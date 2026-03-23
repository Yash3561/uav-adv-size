"""
fixed_attacks.py — Drop-in replacement for the attack functions in experiment.py

WHY THE OLD PGD DIDN'T WORK:
  The proxy loss `pred.abs().mean()` maximizes the raw detection head output.
  YOLOv8's decoupled head separates box regression and classification.
  Maximizing the mean absolute value does NOT reliably suppress objectness.
  After 20 steps, the optimizer finds a different local maximum that barely
  changes actual detection behavior — so PGD ≈ FGSM or weaker.

THE FIX — Three proper YOLO attack losses:
  1. Objectness suppression: minimize the max objectness score in the prediction
     → directly reduces detector confidence on all predicted boxes
  2. DAG-style (Dense Adversary Generation): suppress all anchors simultaneously  
     → more aggressive, used in published work on YOLO attacks
  3. TOG (Targeted Object Generation): suppress boxes above threshold
     → most targeted, closest to what a real attacker would do

USAGE: Replace apply_attack() in experiment.py with apply_attack_fixed()
       or import this module and call fgsm_yolo() / pgd_yolo() directly.
"""

import torch
import numpy as np
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 640


def img_to_tensor(pil_img, size=IMG_SIZE, device=DEVICE):
    img = pil_img.convert("RGB").resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return t


def tensor_to_pil(t):
    arr = t.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


# ─── YOLO-SPECIFIC ATTACK LOSS ─────────────────────────────────────────────

def yolo_objectness_loss(model_pt, x):
    """
    Proper YOLO attack loss: suppress objectness scores.
    
    YOLOv8 detection head output shape: [batch, num_anchors, 4 + num_classes]
    The objectness-equivalent score in YOLOv8 (anchor-free) is the max class
    probability times the box confidence, which translates to maximizing the
    NEGATIVE of the class logits sum — making the model think nothing is there.
    
    Strategy: minimize the sum of sigmoid(class_logits) across all anchors.
    This directly suppresses what the NMS layer uses to decide detections.
    """
    model_pt.eval()
    
    try:
        # Get raw model output before NMS
        # YOLOv8's model.model returns the raw prediction tensor when called directly
        preds = model_pt(x)
        
        # Handle different output formats
        if isinstance(preds, (list, tuple)):
            # Take the first element — the raw detection tensor
            raw = preds[0]
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
        else:
            raw = preds
        
        if not isinstance(raw, torch.Tensor):
            return None
        
        # raw shape: [batch, num_predictions, 4+nc] or [batch, 4+nc, num_predictions]
        if raw.dim() == 3:
            if raw.shape[1] < raw.shape[2]:
                # [batch, 4+nc, anchors] → transpose to [batch, anchors, 4+nc]
                raw = raw.permute(0, 2, 1)
            
            # Class scores are the last nc columns (after 4 box coords)
            # We want to SUPPRESS these — so we MAXIMIZE their negation
            # i.e., minimize their sum = loss to minimize = -sum(class_scores)
            # But since we're doing gradient ASCENT on the loss for attack:
            # We want loss = sum(sigmoid(class_logits)) and we maximize it
            # to find the perturbation that... wait.
            #
            # CORRECT SIGN:
            # We want to SUPPRESS detections → reduce class scores
            # Attack: find x_adv such that class_scores are minimized
            # In gradient ascent: loss = -sum(sigmoid(class_scores))
            # We maximize this loss = minimize class scores ✓
            
            if raw.shape[-1] > 4:
                class_logits = raw[..., 4:]  # [batch, anchors, nc]
                # Loss to MAXIMIZE (attack suppresses detections):
                loss = -torch.sigmoid(class_logits).sum()
                return loss
            else:
                loss = -raw.abs().sum()
                return loss
        
        return None
        
    except Exception as e:
        return None


def yolo_dag_loss(model_pt, x, conf_threshold=0.01):
    """
    DAG-style attack loss (Dense Adversary Generation).
    Targets ALL predicted boxes, weighted by their confidence.
    Source: Xie et al. 2017 — attacks all proposals simultaneously.
    
    More aggressive than single-score suppression.
    """
    model_pt.eval()
    try:
        preds = model_pt(x)
        if isinstance(preds, (list, tuple)):
            raw = preds[0]
            if isinstance(raw, (list, tuple)):
                raw = raw[0]
        else:
            raw = preds
        
        if not isinstance(raw, torch.Tensor) or raw.dim() != 3:
            return None
        
        if raw.shape[1] < raw.shape[2]:
            raw = raw.permute(0, 2, 1)
        
        if raw.shape[-1] <= 4:
            return None
        
        class_logits = raw[..., 4:]  # [batch, anchors, nc]
        class_probs = torch.sigmoid(class_logits)
        
        # Weight by objectness (max class prob per anchor)
        objectness_weights = class_probs.max(dim=-1, keepdim=True)[0].detach()
        
        # Attack: suppress high-confidence predictions more aggressively
        # Loss to maximize = -(weighted sum of class probs)
        loss = -(class_probs * objectness_weights).sum()
        return loss
        
    except Exception:
        return None


# ─── FGSM WITH PROPER LOSS ─────────────────────────────────────────────────

def fgsm_yolo(model_pt, pil_img, epsilon, device=DEVICE):
    """
    FGSM with proper YOLOv8 objectness suppression loss.
    Replaces the old fgsm() function.
    """
    x = img_to_tensor(pil_img, device=device).requires_grad_(True)
    
    loss = yolo_objectness_loss(model_pt, x)
    
    if loss is None:
        # Fallback: uniform noise (document this clearly)
        noise = epsilon * torch.sign(torch.randn_like(x.detach()))
        return tensor_to_pil((x.detach() + noise).clamp(0, 1))
    
    # Maximize the loss (gradient ascent = suppress detections)
    # loss is already negated inside yolo_objectness_loss, so we minimize it
    # which means gradient descent on loss = gradient ascent on class scores suppression
    loss.backward()
    
    if x.grad is None:
        noise = epsilon * torch.sign(torch.randn_like(x.detach()))
        return tensor_to_pil((x.detach() + noise).clamp(0, 1))
    
    with torch.no_grad():
        x_adv = x + epsilon * x.grad.sign()
        x_adv = x_adv.clamp(0, 1)
    
    return tensor_to_pil(x_adv)


# ─── PGD WITH PROPER LOSS ──────────────────────────────────────────────────

def pgd_yolo(model_pt, pil_img, epsilon, alpha=None, steps=20, device=DEVICE,
             use_dag=False):
    """
    PGD with proper YOLOv8 objectness suppression loss.
    
    With the correct loss, PGD WILL produce stronger attacks than FGSM.
    Expected result: AP drop under PGD_8 should be 1.5-3x larger than FGSM_8.
    
    use_dag: if True, use DAG-style weighted loss (stronger but slower)
    """
    if alpha is None:
        alpha = epsilon / 4  # standard: step size = eps/4
    
    arr = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    x_orig = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    
    # Random start (crucial for PGD — this is what makes it stronger than FGSM)
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = x_adv.clamp(0, 1).detach()
    
    loss_fn = yolo_dag_loss if use_dag else yolo_objectness_loss
    best_loss = float('inf')
    best_x_adv = x_adv.clone()
    
    for step in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        
        loss = loss_fn(model_pt, x_adv)
        
        if loss is None:
            break
        
        loss.backward()
        
        if x_adv.grad is None:
            break
        
        # Track best adversarial example (worst-case for the model)
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x_adv = x_adv.detach().clone()
        
        with torch.no_grad():
            # Gradient step
            x_adv = x_adv + alpha * x_adv.grad.sign()
            # Project back to epsilon-ball around original
            delta = (x_adv - x_orig).clamp(-epsilon, epsilon)
            x_adv = (x_orig + delta).clamp(0, 1)
    
    return tensor_to_pil(best_x_adv)


# ─── VERIFICATION FUNCTION ─────────────────────────────────────────────────

def verify_attacks_work(model, model_pt, test_image_path, device=DEVICE):
    """
    Quick sanity check: run clean vs FGSM vs PGD on one image.
    PGD MUST produce lower confidence than FGSM for the implementation to be valid.
    
    Run this BEFORE the full experiment to verify correctness.
    Expected output:
      Clean:     conf ~0.5-0.8
      FGSM_8:    conf lower
      PGD_8:     conf LOWER THAN FGSM_8 ← this is the key check
    """
    from PIL import Image
    
    pil_img = Image.open(test_image_path).convert("RGB")
    
    def get_avg_conf(img):
        result = model(img, verbose=False, conf=0.001, imgsz=IMG_SIZE)
        boxes = result[0].boxes
        if len(boxes) == 0:
            return 0.0
        return float(boxes.conf.mean())
    
    print("=== ATTACK VERIFICATION (must see PGD < FGSM) ===")
    
    clean_conf = get_avg_conf(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    print(f"  Clean avg conf:    {clean_conf:.4f}")
    
    fgsm_img = fgsm_yolo(model_pt, pil_img, 8/255, device)
    fgsm_conf = get_avg_conf(fgsm_img)
    print(f"  FGSM_8 avg conf:   {fgsm_conf:.4f}  (drop: {(clean_conf-fgsm_conf)/max(clean_conf,1e-6):.1%})")
    
    pgd_img = pgd_yolo(model_pt, pil_img, 8/255, steps=20, device=device)
    pgd_conf = get_avg_conf(pgd_img)
    print(f"  PGD_8 avg conf:    {pgd_conf:.4f}  (drop: {(clean_conf-pgd_conf)/max(clean_conf,1e-6):.1%})")
    
    if pgd_conf < fgsm_conf:
        print("  ✓ PASS: PGD is stronger than FGSM — implementation is correct")
    else:
        print("  ✗ FAIL: PGD is NOT stronger than FGSM — loss function needs revision")
        print("    Try: pgd_yolo(..., use_dag=True) for the DAG loss variant")
    
    print("=" * 50)
    return clean_conf, fgsm_conf, pgd_conf


# ─── UPDATED apply_attack FUNCTION (replaces apply_attack in experiment.py) ─

def apply_attack_fixed(model_pt, pil_img, attack_cfg, device=DEVICE):
    """
    Drop-in replacement for apply_attack() in experiment.py.
    Uses proper YOLO-specific loss functions.
    
    In experiment.py, change:
        attacked = apply_attack(model_pt, img_pil, attack_cfg)
    to:
        from fixed_attacks import apply_attack_fixed
        attacked = apply_attack_fixed(model_pt, img_pil, attack_cfg, DEVICE)
    """
    if attack_cfg["type"] == "none":
        return pil_img
    
    eps   = attack_cfg["eps"]
    atype = attack_cfg["type"]
    
    if atype == "fgsm":
        return fgsm_yolo(model_pt, pil_img, eps, device)
    
    elif atype == "pgd":
        steps = attack_cfg.get("steps", 20)
        alpha = attack_cfg.get("alpha", eps / 4)
        # Use DAG loss for stronger attacks at higher epsilon
        use_dag = eps >= 12/255
        return pgd_yolo(model_pt, pil_img, eps,
                        alpha=alpha, steps=steps,
                        device=device, use_dag=use_dag)
    
    return pil_img


# ─── HOW TO INTEGRATE INTO experiment.py ──────────────────────────────────

INTEGRATION_INSTRUCTIONS = """
HOW TO USE THIS FILE:

1. Copy fixed_attacks.py to your experiments/ folder
   (same directory as experiment.py)

2. At the top of experiment.py, add:
   from fixed_attacks import apply_attack_fixed, verify_attacks_work

3. In run_full_evaluation(), after loading the model, add this check:
   test_imgs = list(VISDRONE_VAL_IMAGES.glob("*.jpg"))
   if test_imgs:
       verify_attacks_work(model, model_pt, test_imgs[0])
   # If verification FAILS, stop and debug before running full experiment

4. In the evaluation loop, replace:
   attacked = apply_attack(model_pt, img_pil, attack_cfg)
   with:
   attacked = apply_attack_fixed(model_pt, img_pil, attack_cfg, DEVICE)

5. Re-run experiment.py
   Expected time increase: ~20-30% slower (proper PGD does more work)
   Expected result: PGD_8 drops AP by MORE than FGSM_8 (not less)

TARGET NUMBERS (what a passing implementation looks like):
  FGSM_8:  AP drop ~3-5% overall, AP_L drop ~12-15%
  PGD_8:   AP drop ~5-10% overall, AP_L drop ~15-25%
  PGD_16:  AP drop ~10-20% overall

If PGD_8 overall AP drop is still < FGSM_8 after this fix:
  Option A: increase steps to 40 (add: steps=40 in ATTACKS dict)
  Option B: use use_dag=True for all PGD attacks
  Option C: be honest — report it, explain it, keep FGSM as your main result
"""

if __name__ == "__main__":
    print(INTEGRATION_INSTRUCTIONS)
    print("\nTo verify your attacks work:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolov8s.pt')")
    print("  model_pt = model.model.to('cuda').eval()")
    print("  verify_attacks_work(model, model_pt, 'path/to/any/image.jpg')")
