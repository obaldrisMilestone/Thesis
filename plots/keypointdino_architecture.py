"""
KeypointDINO architecture diagram.

Produces:
  keypointdino_arch.png

Forward-pass flow (left → right):
  RGB Image → DINOv2-S (frozen) → Neck → Decoder ×2 → Head → Sigmoid+NMS → Keypoints
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Colour palette ────────────────────────────────────────────────────────────
C_FROZEN   = "#B8D4E8"   # steel blue  — frozen backbone
C_NECK     = "#B8E0B8"   # sage green  — neck
C_DECODER  = "#FFD9A0"   # warm amber  — decoder blocks
C_HEAD     = "#FFB3A3"   # salmon      — heatmap head
C_POST     = "#D8D8D8"   # light grey  — post-processing / inference
C_IO       = "#EDE8FF"   # lavender    — input / output
C_EDGE_FR  = "#3A7BAA"   # frozen edge
C_EDGE_TR  = "#2E6B2E"   # trainable edge

FIG_W, FIG_H = 22, 9
FONT = "DejaVu Sans"


def box(ax, x, y, w, h, facecolor, edgecolor, lw=2, ls="-", zorder=2):
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.12",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=lw, linestyle=ls, zorder=zorder,
    )
    ax.add_patch(p)
    return p


def label(ax, x, y, text, fs=10, fw="bold", color="black", va="center", ha="center"):
    ax.text(x, y, text, fontsize=fs, fontweight=fw, color=color,
            va=va, ha=ha, fontfamily=FONT, zorder=4,
            multialignment="center")


def dim_tag(ax, x, y, text, color="#555555"):
    """Small italic dimension annotation."""
    ax.text(x, y, text, fontsize=8, color=color, style="italic",
            ha="center", va="top", fontfamily=FONT, zorder=4)


def arrow(ax, x1, y, x2, text="", color="#333333", fs=8):
    ax.annotate(
        "", xy=(x2, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
        zorder=3,
    )
    if text:
        ax.text((x1 + x2) / 2, y + 0.18, text, ha="center", va="bottom",
                fontsize=fs, color="#444", fontfamily=FONT, zorder=5)


def badge(ax, x, y, text, facecolor, edgecolor):
    """Small rectangular badge (e.g. FROZEN / TRAINABLE)."""
    ax.text(x, y, text, fontsize=7.5, fontweight="bold",
            color=edgecolor, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=facecolor, ec=edgecolor, lw=1.2),
            fontfamily=FONT, zorder=5)


def sub_box(ax, x, y, w, h, text, color, ec, fs=8.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                       facecolor=color, edgecolor=ec, linewidth=1.2, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, fontsize=fs, ha="center", va="center",
            fontfamily=FONT, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    CY = 4.8   # vertical centre of main flow
    BH = 2.6   # standard box height

    # ── 0. Input ─────────────────────────────────────────────────────────────
    IX, IW = 0.3, 2.0
    box(ax, IX, CY - BH / 2, IW, BH, C_IO, "#6050AA", lw=2)
    label(ax, IX + IW / 2, CY + 0.3, "RGB\nImage", fs=11)
    dim_tag(ax, IX + IW / 2, CY - BH / 2 - 0.1, "[B, 3, 224, 224]")

    arrow(ax, IX + IW, CY, 2.65)

    # ── 1. DINOv2-S Backbone (Frozen) ────────────────────────────────────────
    DX, DW = 2.7, 6.2
    box(ax, DX, CY - BH / 2, DW, BH, C_FROZEN, C_EDGE_FR, lw=2.5, ls="--")
    label(ax, DX + DW / 2, CY + 0.95, "DINOv2-S Backbone", fs=12)

    # Internal sub-boxes
    sub_y = CY - BH / 2 + 0.22
    sub_h = 0.9
    sub_box(ax, DX + 0.2,  sub_y, 1.55, sub_h, "Patch Embed\n14×14 patches", C_FROZEN, C_EDGE_FR)
    ax.annotate("", xy=(DX + 2.1, sub_y + sub_h / 2), xytext=(DX + 1.75, sub_y + sub_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE_FR, lw=1.3), zorder=4)
    sub_box(ax, DX + 2.15, sub_y, 1.8,  sub_h, "12× Transformer\n(Attn + FFN)", C_FROZEN, C_EDGE_FR)
    ax.annotate("", xy=(DX + 4.2, sub_y + sub_h / 2), xytext=(DX + 3.95, sub_y + sub_h / 2),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE_FR, lw=1.3), zorder=4)
    sub_box(ax, DX + 4.25, sub_y, 1.5,  sub_h, "Layer Norm\n+ Reshape", C_FROZEN, C_EDGE_FR)

    badge(ax, DX + 0.65, CY + BH / 2 - 0.28, " ❄  FROZEN ", C_FROZEN, C_EDGE_FR)
    dim_tag(ax, DX + DW / 2, CY - BH / 2 - 0.1, "tokens [B, 256, 384]  →  reshape [B, 384, 16, 16]")

    arrow(ax, DX + DW, CY, 9.3, text="[B, 384, 16, 16]")

    # ── 2. Neck ───────────────────────────────────────────────────────────────
    NX, NW = 9.35, 2.5
    box(ax, NX, CY - BH / 2, NW, BH, C_NECK, C_EDGE_TR, lw=2)
    label(ax, NX + NW / 2, CY + 0.55, "Neck", fs=12)
    sub_box(ax, NX + 0.2, CY - BH / 2 + 0.22, NW - 0.4, 0.9,
            "1×1 Conv  ·  BN  ·  ReLU\n384 → 256 ch", C_NECK, C_EDGE_TR)
    badge(ax, NX + NW - 0.65, CY + BH / 2 - 0.28, "TRAINABLE", C_NECK, C_EDGE_TR)
    dim_tag(ax, NX + NW / 2, CY - BH / 2 - 0.1, "[B, 256, 16, 16]")

    arrow(ax, NX + NW, CY, 12.1, text="[B, 256, 16, 16]")

    # ── 3. Decoder (two UpsampleBlocks) ──────────────────────────────────────
    DCX, DCW = 12.15, 5.4
    # Outer container
    box(ax, DCX, CY - BH / 2, DCW, BH, "#FFF6E0", C_EDGE_TR, lw=2, ls="-")
    label(ax, DCX + DCW / 2, CY + 0.95, "Decoder", fs=12)

    sub_h2 = 0.9
    sub_y2 = CY - BH / 2 + 0.22
    sub_bw = 2.3
    sub_box(ax, DCX + 0.15, sub_y2, sub_bw, sub_h2,
            "UpsampleBlock 1\n2× Bilinear · Conv 3×3 · BN · ReLU\n256 → 128 ch", C_DECODER, C_EDGE_TR)
    ax.annotate("", xy=(DCX + sub_bw + 0.55, sub_y2 + sub_h2 / 2),
                xytext=(DCX + sub_bw + 0.15, sub_y2 + sub_h2 / 2),
                arrowprops=dict(arrowstyle="-|>", color=C_EDGE_TR, lw=1.3), zorder=4)
    ax.text(DCX + sub_bw + 0.35, sub_y2 + sub_h2 / 2 + 0.12,
            "32×32", fontsize=7.5, ha="center", color="#444", fontfamily=FONT)
    sub_box(ax, DCX + sub_bw + 0.6, sub_y2, sub_bw, sub_h2,
            "UpsampleBlock 2\n2× Bilinear · Conv 3×3 · BN · ReLU\n128 → 64 ch", C_DECODER, C_EDGE_TR)

    badge(ax, DCX + 0.7, CY + BH / 2 - 0.28, "TRAINABLE", C_DECODER, C_EDGE_TR)
    dim_tag(ax, DCX + DCW / 2, CY - BH / 2 - 0.1, "[B, 64, 64, 64]")

    arrow(ax, DCX + DCW, CY, 17.9, text="[B, 64, 64, 64]")

    # ── 4. Heatmap Head ───────────────────────────────────────────────────────
    HX, HW = 17.95, 2.2
    box(ax, HX, CY - BH / 2, HW, BH, C_HEAD, "#A03020", lw=2)
    label(ax, HX + HW / 2, CY + 0.55, "Head", fs=12)
    sub_box(ax, HX + 0.2, CY - BH / 2 + 0.22, HW - 0.4, 0.9,
            "1×1 Conv\n64 → 1 ch  (logits)", C_HEAD, "#A03020")
    badge(ax, HX + HW - 0.65, CY + BH / 2 - 0.28, "TRAINABLE", C_HEAD, "#A03020")
    dim_tag(ax, HX + HW / 2, CY - BH / 2 - 0.1, "[B, 1, 64, 64]")

    # ── Arrow down then right to inference section ────────────────────────────
    # Vertical drop to inference row
    INF_Y = 1.5
    ax.annotate("", xy=(HX + HW / 2, INF_Y + 1.0),
                xytext=(HX + HW / 2, CY - BH / 2),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.8), zorder=3)
    ax.text(HX + HW / 2 + 0.18, (CY - BH / 2 + INF_Y + 1.0) / 2,
            "inference only", fontsize=7.5, color="#888", rotation=90, va="center")

    # ── 5. Sigmoid + NMS (inference) ─────────────────────────────────────────
    SX, SW = 16.5, 3.4
    SY = INF_Y - 0.15
    SH = 1.4
    box(ax, SX, SY, SW, SH, C_POST, "#888888", lw=1.5, ls="--")
    label(ax, SX + SW / 2, SY + SH / 2 + 0.18, "Post-Processing", fs=10)
    ax.text(SX + SW / 2, SY + SH / 2 - 0.22,
            "Sigmoid  ·  MaxPool NMS (3×3)  ·  scale to image coords",
            fontsize=8.5, ha="center", va="center", fontfamily=FONT, color="#333")

    arrow(ax, SX, SY + SH / 2, 15.5, color="#555")

    # ── 6. Output Keypoints ───────────────────────────────────────────────────
    OX, OW = 13.3, 2.0
    OY = SY
    box(ax, OX, OY, OW, SH, C_IO, "#6050AA", lw=2)
    label(ax, OX + OW / 2, OY + SH / 2 + 0.15, "Keypoints", fs=10.5)
    ax.text(OX + OW / 2, OY + SH / 2 - 0.22,
            "(x, y) in image coords",
            fontsize=8, ha="center", va="center", fontfamily=FONT, color="#333")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_x, legend_y = 0.3, 1.9
    items = [
        (C_FROZEN, C_EDGE_FR, "--", "Frozen (DINOv2-S)"),
        (C_NECK,   C_EDGE_TR, "-",  "Trainable — Neck"),
        (C_DECODER,C_EDGE_TR, "-",  "Trainable — Decoder"),
        (C_HEAD,   "#A03020", "-",  "Trainable — Head"),
        (C_POST,   "#888888", "--", "Post-processing (inference)"),
    ]
    for k, (fc, ec, ls, txt) in enumerate(items):
        lx = legend_x + k * 4.0
        p = FancyBboxPatch((lx, legend_y), 0.6, 0.45,
                           boxstyle="round,pad=0.05",
                           facecolor=fc, edgecolor=ec, linewidth=1.5, linestyle=ls, zorder=3)
        ax.add_patch(p)
        ax.text(lx + 0.75, legend_y + 0.22, txt, fontsize=8.5,
                va="center", fontfamily=FONT, color="#222")

    # ── Title ────────────────────────────────────────────────────────────────
    ax.text(FIG_W / 2, FIG_H - 0.45, "KeypointDINO — RGB Corner Detection Architecture",
            ha="center", va="top", fontsize=15, fontweight="bold",
            fontfamily=FONT, color="#111")

    plt.tight_layout(pad=0)
    out = os.path.join(OUTPUT_DIR, "keypointdino_arch.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
