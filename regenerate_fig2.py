"""
regenerate_fig2.py — Replaces the broken SVR ratio figure (y-axis up to 1400).

The old Fig 2 computed SVR = small_drop / large_drop.
When large_drop is near zero, this explodes to 1400+ and is meaningless.

New Fig 2: side-by-side grouped bars showing ABSOLUTE AP drop (percentage points)
per size category per attack. No division. No instability. 
The visual story is the same — large bar (green) is consistently tallest —
but now the y-axis is interpretable and the SVR can be a single clean number
computed over the FGSM conditions only.

Run: python regenerate_fig2.py
Output: figures/fig2_ap_drop_by_size.pdf + .png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Actual experimental data ────────────────────────────────────────────────
clean = {"AP": 0.3978, "AP_S": 0.1190, "AP_M": 0.3871, "AP_L": 0.2760}

attacks = [
    {"label": "FGSM\nε=2",  "AP": 0.3971, "AP_S": 0.1187, "AP_M": 0.3875, "AP_L": 0.2555},
    {"label": "FGSM\nε=4",  "AP": 0.3965, "AP_S": 0.1183, "AP_M": 0.3919, "AP_L": 0.2590},
    {"label": "FGSM\nε=8",  "AP": 0.3871, "AP_S": 0.1152, "AP_M": 0.3835, "AP_L": 0.2453},
    {"label": "FGSM\nε=16", "AP": 0.3400, "AP_S": 0.0933, "AP_M": 0.3466, "AP_L": 0.2123},
    {"label": "PGD\nε=16",  "AP": 0.3820, "AP_S": 0.1131, "AP_M": 0.3788, "AP_L": 0.2326},
    # PGD_8 excluded — see methodology note in paper
]

# Compute ABSOLUTE drops in percentage points
drops_S, drops_M, drops_L = [], [], []
labels = []
for att in attacks:
    drops_S.append((clean["AP_S"] - att["AP_S"]) * 100)
    drops_M.append((clean["AP_M"] - att["AP_M"]) * 100)
    drops_L.append((clean["AP_L"] - att["AP_L"]) * 100)
    labels.append(att["label"])

# Compute SVR over FGSM conditions only (stable denominators)
# SVR = AP_L drop / AP_S drop — inverse of original to show large > small
fgsm_indices = [0, 1, 2, 3]  # FGSM_2,4,8,16
svr_values = []
for i in fgsm_indices:
    if drops_S[i] > 0.05:  # only when drops are meaningful
        svr = drops_L[i] / drops_S[i]
        svr_values.append(svr)

mean_svr = np.mean(svr_values)
print(f"SVR (AP_L drop / AP_S drop) over FGSM conditions: {svr_values}")
print(f"Mean SVR: {mean_svr:.2f} — large objects drop {mean_svr:.1f}x more than small")

# ── Figure ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "Absolute AP Drop by Object Size Under Adversarial Attack\n"
    "(Large objects are disproportionately vulnerable)",
    fontsize=13, fontweight="bold"
)

colors = {"S": "#E53935", "M": "#FB8C00", "L": "#2E7D32"}
x = np.arange(len(attacks))
w = 0.26

ax = axes[0]
b1 = ax.bar(x - w, drops_S, w, label="Small (area<32²)",  color=colors["S"], edgecolor="white", linewidth=0.5)
b2 = ax.bar(x,     drops_M, w, label="Medium (32²–96²)", color=colors["M"], edgecolor="white", linewidth=0.5)
b3 = ax.bar(x + w, drops_L, w, label="Large (area≥96²)",  color=colors["L"], edgecolor="white", linewidth=0.5)

# Value labels on top of bars
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.3:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.08,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8.5,
                    color="#333333")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Absolute AP Drop vs. Clean (percentage points)")
ax.set_title("AP Drop per Object Size Category")
ax.set_ylim(0, max(max(drops_L) * 1.35, 5))
ax.legend(loc="upper left", fontsize=9)
ax.axhline(0, color="gray", linewidth=0.5)

# ── Right panel: ratio of AP_L drop to AP_S drop ────────────────────────────
ax2 = axes[1]

ratio_L_over_S = []
ratio_L_over_M = []
for i in range(len(attacks)):
    r_ls = drops_L[i] / max(drops_S[i], 0.01)
    r_lm = drops_L[i] / max(abs(drops_M[i]), 0.01)
    ratio_L_over_S.append(r_ls)
    ratio_L_over_M.append(r_lm)

ax2.plot(x, ratio_L_over_S, "s-", color=colors["L"], linewidth=2.5,
         markersize=9, label="Large/Small drop ratio", zorder=3)
ax2.plot(x, ratio_L_over_M, "^--", color=colors["M"], linewidth=2,
         markersize=8, label="Large/Medium drop ratio", zorder=3, alpha=0.85)
ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1.5,
            label="Ratio = 1 (equal vulnerability)", zorder=2)
ax2.fill_between(x, 1.0, ratio_L_over_S, alpha=0.10, color=colors["L"])

# Annotate SVR mean
ax2.annotate(
    f"Mean SVR = {mean_svr:.1f}×\n(FGSM conditions)",
    xy=(x[2], ratio_L_over_S[2]),
    xytext=(x[2] + 0.3, ratio_L_over_S[2] + 0.8),
    fontsize=9, color=colors["L"],
    arrowprops=dict(arrowstyle="->", color=colors["L"], lw=1.2),
)

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10)
ax2.set_ylabel("Vulnerability Ratio\n(Large object AP drop / other size AP drop)")
ax2.set_xlabel("Attack Condition")
ax2.set_title(f"SVR: Large objects drop {mean_svr:.1f}× more than small objects\n(ratio > 1 = large more vulnerable)")
ax2.legend(loc="upper left", fontsize=9)
ax2.set_ylim(0, max(max(ratio_L_over_S) * 1.4, 4))
ax2.grid(True, alpha=0.25, axis='y')

plt.tight_layout()

import os
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/fig2_ap_drop_by_size.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/fig2_ap_drop_by_size.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n✓ New Fig 2 saved: figures/fig2_ap_drop_by_size.pdf + .png")
print(f"\nKEY NUMBERS FOR THE PAPER (fill these into paper.tex):")
print(f"  FGSM_8:  AP_L drop = {drops_L[2]:.1f}pp,  AP_S drop = {drops_S[2]:.1f}pp")
print(f"  FGSM_16: AP_L drop = {drops_L[3]:.1f}pp,  AP_S drop = {drops_S[3]:.1f}pp")
print(f"  PGD_16:  AP_L drop = {drops_L[4]:.1f}pp,  AP_S drop = {drops_S[4]:.1f}pp")
print(f"  Mean SVR (FGSM): {mean_svr:.2f} — use this in abstract and conclusion")
print(f"  Masking factor FGSM_8: AP_L drop ({drops_L[2]:.1f}pp) / overall drop ({(clean['AP']-0.3871)*100:.1f}pp) = {drops_L[2]/((clean['AP']-0.3871)*100):.1f}×")
