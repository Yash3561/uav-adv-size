# Adversarial Vulnerability Is Not Uniform: Large-Object Detection Suffers Disproportionate Degradation in UAV Imagery

**Yash Chaudhary, Chengjun Liu**
Department of Computer Science, New Jersey Institute of Technology

---

## Overview

Standard adversarial robustness evaluations of object detectors report aggregate mAP, which is dominated by small objects (71% of VisDrone instances). This masks a critical finding:

> **Large objects suffer 11.6× more relative AP degradation than small objects under adversarial attack.**

Under FGSM at ε=8/255, AP_L drops 11.1% relative to clean while AP_S drops only 3.2%. Aggregate mAP underestimates large-object vulnerability by **4.1×** because the numerically dominant small-object category undergoes minimal degradation.

We introduce the **Size-Conditioned Vulnerability Ratio (SVR)** — a lightweight metric that directly quantifies adversarial vulnerability asymmetry across COCO-standard object size categories.

---

## Results

| Attack | AP | AP_S | AP_M | AP_L | SVR (L/S) |
|---|---|---|---|---|---|
| Clean | 39.8% | 11.9% | 38.7% | 27.6% | — |
| FGSM ε=2 | 39.7% | 11.9% | 38.8% | 25.6% | 24.3× |
| FGSM ε=4 | 39.6% | 11.8% | 39.2% | 25.9% | 8.1× |
| FGSM ε=8 | 38.7% | 11.5% | 38.4% | 24.5% | 3.5× |
| FGSM ε=16 | 34.0% | 9.3% | 34.7% | 21.2% | 1.1× |
| PGD ε=16 | 38.2% | 11.3% | 37.9% | 23.3% | 2.8× |

Mean SVR (FGSM conditions) = **11.6×**

---

## Repository Structure

```
uav-adv-size/
├── REAL_paper.tex          # Full LaTeX paper (IEEE two-column)
├── REAL_experiment.ipynb   # Complete experiment notebook
├── REAL_README.md          # Detailed experiment notes
├── fixed_attacks.py        # Corrected PGD attack implementation
├── regenerate_fig2.py      # Script to regenerate Figure 2
├── figures/
│   ├── fig1_ap_by_size.pdf/.png        # AP by size under each attack
│   ├── fig2_ap_drop_by_size.pdf/.png   # Absolute AP drop + SVR ratio
│   ├── fig3_ap_heatmap.pdf/.png        # AP heatmap across all conditions
│   └── fig4_dataset_and_baseline.pdf/.png  # Dataset distribution + clean AP
└── results/
    ├── ap_by_size.json         # Raw numerical results
    └── table1_ap_by_size.tex   # LaTeX table (included in paper)
```

---

## Setup

### Requirements

```bash
pip install ultralytics torch torchvision numpy matplotlib
```

Tested with Python 3.10+, PyTorch 2.x, Ultralytics 8.x.

### Dataset

Download **VisDrone-DET2019** validation split from the [official VisDrone benchmark](https://github.com/VisDrone/VisDrone-Dataset) and place it at:

```
data/VisDrone2019-DET-val/
├── images/
└── annotations/
```

### Model weights

YOLOv8s COCO-pretrained weights are downloaded automatically by Ultralytics on first run, or manually:

```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
```

---

## Running the Experiment

Open `REAL_experiment.ipynb` and run all cells. The notebook:

1. Loads YOLOv8s with COCO-pretrained weights
2. Evaluates on the VisDrone-DET2019 val split (300 images) under clean conditions
3. Applies FGSM (ε ∈ {2,4,8,16}/255) and PGD (ε ∈ {16}/255) attacks
4. Computes AP_S, AP_M, AP_L, and aggregate AP at IoU=0.5 for each condition
5. Saves results to `results/ap_by_size.json`

To regenerate Figure 2 from existing results:

```bash
python regenerate_fig2.py
```

---

## SVR Metric

The **Size-Conditioned Vulnerability Ratio** is defined as:

$$\text{SVR}_{S/L} = \frac{\Delta\text{AP}_S / \text{AP}_S^{\text{clean}}}{\Delta\text{AP}_L / \text{AP}_L^{\text{clean}}}$$

- SVR < 1 → large objects are disproportionately more vulnerable
- SVR = 1 → equal proportional degradation across sizes
- SVR > 1 → small objects are disproportionately more vulnerable

SVR < 1 in all evaluated conditions.

---

## Citation

If you use this work, please cite:

```bibtex
@article{chaudhary2025adversarial,
  title={Adversarial Vulnerability Is Not Uniform: Large-Object Detection
         Suffers Disproportionate Degradation in UAV Imagery Under Adversarial Attack},
  author={Chaudhary, Yash and Liu, Chengjun},
  journal={arXiv preprint},
  year={2025}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
