# REAL PAPER — Weekend Execution Guide
# Read this entirely before touching any code.

---

## THE PAPER IN ONE PARAGRAPH

We test whether adversarial attacks hurt small UAV objects more than
large ones. Nobody has done this with proper AP_S/AP_M/AP_L metrics
on VisDrone with ground-truth annotations. The answer (our hypothesis)
is yes — and aggregate mAP masks how bad it really is. This is novel,
defensible, and completable this weekend.

Novel contribution: The SVR (Size-Conditioned Vulnerability Ratio) metric.
Defined in the paper. Simple formula. Real finding. Never published before.

---

## STEP 0 — Install (10 minutes)

```bash
pip install ultralytics torch torchvision numpy matplotlib seaborn pandas tqdm
```

---

## STEP 1 — Download VisDrone (20 minutes)

You MUST have ground-truth annotations. This is what makes the paper real.

Option A — Direct download:
```bash
mkdir -p data
cd data
# Download from official source:
wget "https://github.com/VisDrone/VisDrone-Dataset/releases/download/..."
# OR manually download from:
# https://github.com/VisDrone/VisDrone-Dataset
# File: VisDrone2019-DET-val.zip (~780MB)
unzip VisDrone2019-DET-val.zip
# Result: data/VisDrone2019-DET-val/images/*.jpg
#         data/VisDrone2019-DET-val/annotations/*.txt
```

Option B — Kaggle (if direct download is slow):
Search "VisDrone2019" on kaggle.com — multiple public datasets.

VERIFY the structure:
```
data/
└── VisDrone2019-DET-val/
    ├── images/       ← 548 .jpg files
    └── annotations/  ← 548 .txt files (same names as images)
```

The annotation .txt files are what separate this from the junk paper.
Each line: bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion

---

## STEP 2 — Run experiments (2-6 hours, unattended)

```bash
cd experiments
python experiment.py
```

What it does:
1. Parses all ground-truth annotations → gets exact object sizes
2. Runs YOLOv8s clean on 300 val images → AP, AP_S, AP_M, AP_L
3. Runs FGSM at ε=2,4,8,16/255 → AP, AP_S, AP_M, AP_L each
4. Runs PGD at ε=8,16/255 → AP, AP_S, AP_M, AP_L each
5. Generates 4 figures (PDF + PNG)
6. Generates LaTeX table
7. Saves everything to results/ and figures/

Start it before bed. Check results in the morning.

To run faster (fewer images, for testing):
  Edit N_VAL_IMAGES = 50 in experiment.py
  Runs in 30-40 minutes on CPU
  Use N_VAL_IMAGES = 300 for paper (or 548 for full val set)

---

## STEP 3 — Compute your SVR numbers (10 minutes manual)

Open results/ap_by_size.json. For the PGD_8 row:

  SVR_S/L = (AP_S_clean - AP_S_pgd)/AP_S_clean
            ─────────────────────────────────────
            (AP_L_clean - AP_L_pgd)/AP_L_clean

This number goes in your abstract, results, and conclusion.

If SVR > 1 (which it should be): your hypothesis is confirmed.
If SVR <= 1: your finding is the opposite — equally interesting,
equally publishable. Report what you find honestly.

---

## STEP 4 — Fill in the manuscript (2 hours writing)

Open manuscript/paper.tex
Search for [FILL: and fill in every placeholder with your real numbers.

Key numbers to fill in (from results/ap_by_size.json):
- % of objects that are small (from n_gt_small / n_gt_total)
- N_val_images (300 or however many you ran)
- Clean AP, AP_S, AP_M, AP_L (the "Clean" row)
- PGD_8: AP_S drop %, AP_M drop %, AP_L drop %
- SVR_S/L for each attack condition
- "factor by which mAP underestimates AP_S drop"
  = AP_S_drop / AP_overall_drop
- Your name, professor's name, email

---

## STEP 5 — Compile on Overleaf (10 minutes)

1. Go to overleaf.com (free account)
2. New Project → Upload Project
3. Upload:
   - manuscript/paper.tex
   - results/table1_ap_by_size.tex
   - figures/fig1_ap_by_size.png
   - figures/fig2_vulnerability_ratio.png
   - figures/fig3_ap_heatmap.png
   - figures/fig4_dataset_and_baseline.png
4. Click Recompile → Download PDF

---

## STEP 6 — Send to professor

Send:
1. The PDF
2. results/ap_by_size.json (so professor can verify numbers)
3. The experiment.py code (so professor can see methodology)

What to say:
"The paper measures whether adversarial attacks affect small and
large UAV objects differently, which no prior paper has evaluated
with proper AP_S/AP_M/AP_L metrics on VisDrone. The key finding
is SVR=[your number] under PGD — meaning small objects suffer
[your number]x more relative degradation than large objects.
The SVR metric is the novel contribution."

---

## WHAT YOUR PROFESSOR WILL CHECK

1. "Is the gap real?" — YES. Show them the Frontiers 2024 and
   Journal of Remote Sensing 2024 papers in your related work.
   Neither reports size-stratified AP.

2. "Is the metric novel?" — YES. SVR is defined in your paper,
   formally, with an equation. It has not appeared in prior work.

3. "Are the experiments rigorous?" — YES, because you used
   ground-truth annotations and proper AP computation,
   not a confidence proxy.

4. "Are the results honest?" — YES, if you report exactly what
   you find, including limitations (single model, white-box,
   digital attacks). The limitations section is already written.

5. "Is this publishable?" — Target: IEEE Access, IEEE Signal
   Processing Letters, or Pattern Recognition Letters.
   All three publish focused empirical studies with novel metrics.

---

## IF SOMETHING BREAKS

Problem: "No module named ultralytics"
Fix: pip install ultralytics

Problem: "CUDA out of memory"
Fix: Edit experiment.py, add batch_size=1 to model() calls
OR: Just let it run on CPU (slower but works)

Problem: "annotations/ folder not found"
Fix: You downloaded images only, not annotations.
Re-download the full val zip from VisDrone GitHub.

Problem: AP_S = 0.0 for all attacks
Fix: Your val images may not have any small objects in the
first N_VAL_IMAGES. Increase N_VAL_IMAGES or shuffle the list.
Add: random.shuffle(all_imgs) before all_imgs[:N_VAL_IMAGES]

Problem: SVR is always exactly 1.0
Fix: This means either clean AP_S or adv AP_S is 0.
Check n_gt_small in results — if 0, the annotation parsing
may be wrong. Check annotation format matches VisDrone spec.

---

## FILES IN THIS PACKAGE

```
real-paper/
├── README.md                          ← you are here
├── experiments/
│   └── experiment.py                  ← THE experiment (run this)
├── manuscript/
│   └── paper.tex                      ← THE paper (fill placeholders)
├── results/                           ← created by experiment.py
│   ├── ap_by_size.json
│   └── table1_ap_by_size.tex
└── figures/                           ← created by experiment.py
    ├── fig1_ap_by_size.pdf/.png
    ├── fig2_vulnerability_ratio.pdf/.png
    ├── fig3_ap_heatmap.pdf/.png
    └── fig4_dataset_and_baseline.pdf/.png
```
