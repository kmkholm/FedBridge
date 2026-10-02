"""Methodology figure (glyph style of the TRACE-Mamba v2 figure, new palette).
Seven panels: evidence and rules, behaviour tensors and USD units, protocols and account corpus,
BiMamba block zoom-in, pooling and KAN head, neuro-symbolic triage, federation and explanation.
Run: python gen_methodology.py -> methodology.drawio (+ PDF/PNG via draw.io CLI)
"""
import math, os, subprocess, sys
from figlib import Fig, FONT

f = Fig(page_w=1560, page_h=1020, scale=0.6, name="Bridge triage methodology")
TITLE, PSTROKE = "#7a1f3d", "#b0848f"
NAVY = ["#f2f5fa", "#dce4f1", "#b9c9e4", "#8ea8d2", "#6585bd", "#43649f", "#2a467a"]
CRIM = ["#fff2f0", "#fdd8d2", "#f8ab9f", "#ee7d6e", "#d9534f", "#b03a35", "#7a2421"]
GOLD = ["#fffbea", "#fdf1c7", "#f9df8e", "#f2c75a", "#dca72c", "#b0811c", "#765612"]
JADE = ["#effaf4", "#d2f0e0", "#a6e0c4", "#70c8a1", "#40ab7e", "#258a60", "#155d40"]
PLUM = ["#f7f2fb", "#e8dcf3", "#d0b8e6", "#b293d5", "#9270c0", "#7150a3", "#4e3378"]
EVT = ["#8ea8d2", "#f2c75a", "#70c8a1", "#b293d5", "#ee7d6e", "#9fb3c8", "#f9df8e", "#a6e0c4", "#d0b8e6"]


def panel(x, y, w, h, title, fill, fs=30):
    f.rect(x, y, w, h, "", fill, PSTROKE, arc=3, sw=1.6, extra="dashed=1;dashPattern=8 5;")
    f.text(x + 8, y + 6, w - 16, 48, title, fs=fs, color=TITLE, extra="fontStyle=3;whiteSpace=nowrap;")


def lab(x, y, w, h, s, fs=18, align="center", color="#000000", style=""):
    return f.text(x, y, w, h, s, fs=fs, align=align, color=color, extra="whiteSpace=nowrap;" + style)


def wlab(x, y, w, h, s, fs=14, align="center", color="#444444"):
    return f.text(x, y, w, h, s, fs=fs, align=align, color=color, extra="whiteSpace=wrap;")


def cube(x, y, w, h, fill, stroke, depth=10, label="", fs=15):
    st = (f"shape=cube;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;darkOpacity=0.06;darkOpacity2=0.14;"
          f"size={depth};fillColor={fill};strokeColor={stroke};strokeWidth=1.3;fontSize={fs};" + FONT)
    return f.vertex(label, st, x, y, w, h)


def cyl(x, y, w, h, fill, stroke, label="", fs=17, color="#ffffff"):
    st = (f"shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=11;fillColor={fill};"
          f"strokeColor={stroke};strokeWidth=1.4;fontSize={fs};fontStyle=1;fontColor={color};" + FONT)
    return f.vertex(label, st, x, y, w, h)


def doc(x, y, w, h, fill, stroke, label="", fs=18):
    st = (f"shape=note;size=18;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=1.4;"
          f"fontSize={fs};fontStyle=1;verticalAlign=bottom;spacingBottom=4;" + FONT)
    f.vertex(label, st, x, y, w, h)


def node(cx, cy, r, fill="#ffffff", stroke="#333333", label="", fs=16):
    f.circle(cx, cy, r, label, fill=fill, stroke=stroke, fs=fs)


def op(cx, cy, sym, r=19, fill="#ffffff", fs=21):
    node(cx, cy, r, fill, "#333333", sym, fs)


def strip(x, y, n, c, pal, vertical=False, seed=0.0, stroke="#7a7a7a"):
    for k in range(n):
        v = int((math.sin(k * 1.7 + seed) + 1) / 2 * (len(pal) - 3)) + 2
        xx, yy = (x, y + k * c) if vertical else (x + k * c, y)
        f.rect(xx, yy, c, c, "", pal[v], stroke, arc=0, sw=0.8)


def matrix(x, y, rows, cols, c, pal, seed=0.0, pad_rows=0, stroke="#9a9a9a"):
    for i in range(rows):
        for j in range(cols):
            if i < pad_rows:
                f.rect(x + j * c, y + i * c, c, c, "", "#ffffff", "#dddddd", arc=0, sw=0.4); continue
            v = int((math.sin(i * 1.3 + j * 2.1 + seed) + 1) / 2 * (len(pal) - 3)) + 2
            f.rect(x + j * c, y + i * c, c, c, "", pal[v], stroke, arc=0, sw=0.5)


def curve_icon(x, y, w, h, fn, fill, stroke, col, x0=-4, x1=4):
    f.rect(x, y, w, h, "", fill, stroke, arc=18, sw=1.2)
    vals = [fn(x0 + (x1 - x0) * k / 12) for k in range(13)]
    lo, hi = min(vals), max(vals)
    pts = [(x + 6 + (w - 12) * k / 12, y + h - 7 - (h - 14) * (v - lo) / (hi - lo + 1e-9)) for k, v in enumerate(vals)]
    f.arrow(pts, head=False, curved=True, color=col, sw=2)


def rows_list(x, y, n, w, h, hot, gap=4, cold=NAVY[1], hotc=CRIM[4]):
    for k in range(n):
        f.rect(x, y + k * (h + gap), w, h, "", hotc if k in hot else cold, "#8a8a8a", arc=20, sw=0.7)


silu = lambda t: t / (1 + math.exp(-t))
softplus = lambda t: math.log1p(math.exp(t))

# ======================================================== P1 evidence and conservation rules
panel(15, 15, 885, 560, "① Evidence & Conservation Rules", "#f5f7fc", fs=26)
cyl(45, 92, 112, 112, NAVY[5], NAVY[6], "Ethe-<br>reum", fs=14)
f.rect(200, 118, 158, 62, "🔒 bridge vault", GOLD[2], GOLD[5], fs=16, arc=16, sw=1.4)
cyl(400, 92, 112, 112, PLUM[4], PLUM[6], "Ronin<br>Moon-<br>beam", fs=13)
f.arrow([(159, 149), (197, 149)], sw=1.6); f.arrow([(360, 149), (397, 149)], sw=1.6)
lab(195, 184, 170, 22, "lock <i>d</i>  →  release <i>w</i>", fs=13, color="#555555")
lab(40, 214, 480, 24, "187 k tx · 1.23 M events · 9 Datalog relations", fs=13, color="#555555")
# onset chart (label refinement)
bx, by, bw = 560, 92, 300
f.rect(bx, by, bw, 128, "", "#ffffff", "#c8c8c8", arc=4, sw=0.8)
hts = [3, 5, 4, 6, 8, 4, 23, 161, 34, 20, 12, 8]
for k, v in enumerate(hts):
    hh = 6 + 100 * math.sqrt(v / 161)
    f.rect(bx + 14 + k * 23, by + 118 - hh, 17, hh, "", CRIM[4] if k >= 6 else NAVY[3], "#7a7a7a", arc=0, sw=0.5)
f.arrow([(bx + 151, by + 6), (bx + 151, by + 122)], head=False, dashed=True, color=CRIM[6], sw=1.3)
lab(bx, by - 2, bw, 20, "onset 21:00 UTC", fs=12, color=CRIM[6])
lab(bx - 5, by + 130, bw + 10, 22, "Nomad, 1 Aug 2022 (hourly)", fs=12, color="#555555")
lab(bx, by + 152, bw, 22, "465 = 2 Ronin + 463 Nomad", fs=13, color=CRIM[6], style="fontStyle=1;")
# deposit / release matching
lab(55, 268, 130, 24, "<i>d</i>  deposits", fs=15, color=NAVY[6])
lab(255, 268, 140, 24, "<i>w</i>  releases", fs=15, color=GOLD[6])
D = [(85, 300 + k * 30) for k in range(7)]
W = [(275, 300 + k * 30) for k in range(7)]
for (x, y) in D:
    f.rect(x, y, 70, 22, "", NAVY[2], NAVY[5], arc=20, sw=0.9)
for k, (x, y) in enumerate(W):
    f.rect(x, y, 70, 22, "", CRIM[3] if k >= 5 else GOLD[2], CRIM[6] if k >= 5 else GOLD[5], arc=20, sw=1.0)
for a, b in [(0, 0), (1, 2), (2, 1), (3, 3), (4, 4)]:
    f.arrow([(157, D[a][1] + 11), (273, W[b][1] + 11)], head=False, color=JADE[5], sw=1.3)
for k in (5, 6):
    lab(350, W[k][1] - 3, 30, 26, "✗", fs=18, color=CRIM[6], style="fontStyle=1;")
lab(40, 515, 380, 30, "<i>w</i> ⇒ ∃ <i>d</i> : match(<i>d</i>, <i>w</i>)", fs=18, color=JADE[6])
f.arrow([(380, 466), (440, 466)], sw=1.6, color=CRIM[5])
# alert queue
f.vertex("🔔", "text;html=1;fontSize=30;align=center;verticalAlign=middle;", 445, 268, 46, 46)
lab(495, 276, 300, 30, "alert queue 𝒜", fs=19, align="left", style="fontStyle=1;")
rows_list(470, 322, 9, 390, 16, hot={5})
lab(455, 520, 420, 26, "16,396 alerts · 465 exploits · 34 : 1 noise", fs=15)

# ======================================================== P2 behaviour tensors and economic units
panel(915, 15, 885, 560, "② Behaviour Tensors & Economic Units", "#fdfaf1", fs=26)
icons = ["$", "#", "Σ", "◷", "Δ"]
mx, my, c = 975, 112, 20
for j, ic in enumerate(icons):
    lab(mx + j * c - 2, my - 26, c + 4, 24, ic, fs=13)
for j in range(9):
    f.rect(mx + (5 + j) * c + 4, my - 18, 12, 12, "", EVT[j], "#777777", arc=0, sw=0.5)
lab(mx + 5 * c, my - 46, 9 * c, 20, "event types", fs=11, color="#666666")
matrix(mx, my, 11, 14, c, NAVY, seed=0.4, pad_rows=3)
f.arrow([(955, my), (955, my + 220)], sw=1.3)
lab(930, my + 95, 24, 26, "<i>t</i>", fs=17)
lab(mx + 14 * c + 3, my + 18, 50, 22, "pad", fs=12, color="#777777", style="align=left;")
lab(950, my + 228, 340, 26, "<b>x</b> ∈ ℝ<sup>32×14</sup> · sender history", fs=15)
# context sparkline
cx0, cy0 = 1335, 92
f.rect(cx0, cy0, 250, 160, "", "#ffffff", "#c8c8c8", arc=4, sw=0.8)
wr = [0.36, 0.40, 0.35, 0.38, 0.41, 0.37, 0.39, 0.72, 0.80, 0.77, 0.74, 0.70]
for k, v in enumerate(wr):
    hh = 120 * v
    f.rect(cx0 + 14 + k * 20, cy0 + 148 - hh, 14, hh, "", CRIM[3] if k >= 7 else NAVY[3], "#7a7a7a", arc=0, sw=0.5)
lab(cx0 + 4, cy0 + 2, 120, 22, "w-ratio<sub>1h</sub>", fs=13, color="#444444", style="align=left;")
lab(cx0, cy0 + 162, 250, 22, "0.38 benign → 0.76 attack", fs=13, color=CRIM[6])
strip(1625, 92, 10, 16, NAVY, vertical=True, seed=1.1)
lab(1648, 150, 70, 26, "<b>c</b> ∈ ℝ<sup>10</sup>", fs=15, style="align=left;")
wlab(1330, 284, 460, 40, "1 h / 24 h windows: tx and withdrawal counts, withdrawal ratio, Σ amount, new-account share", fs=12)
# USD normalisation
f.rect(960, 380, 150, 40, "USDC · 10<sup>6</sup>", GOLD[1], GOLD[5], fs=16, arc=40, sw=1.3)
f.rect(960, 432, 150, 40, "ETH · 10<sup>18</sup>", PLUM[1], PLUM[5], fs=16, arc=40, sw=1.3)
lab(950, 478, 180, 22, "+ 10 further tokens", fs=12, color="#666666")
f.rect(1170, 402, 200, 52, "÷ 10<sup>dec</sup> × <i>p</i><sub>2022</sub>", "#ffffff", GOLD[5], fs=15, arc=20, sw=1.5)
f.arrow([(1112, 400), (1177, 418)], sw=1.2); f.arrow([(1112, 452), (1177, 440)], sw=1.2)
strip(1395, 416, 6, 24, GOLD, seed=0.7)
lab(1395, 386, 150, 26, "<i>v</i>(<i>a</i>)  in USD", fs=14)
f.arrow([(1372, 428), (1392, 428)], sw=1.2)
f.rect(1570, 406, 200, 44, "log<sub>10</sub>(1 + <i>v</i>)", GOLD[2], GOLD[5], fs=17, arc=20, sw=1.4)
f.arrow([(1540, 428), (1567, 428)], sw=1.2)
lab(1150, 490, 630, 26, "Ronin USDC exploit leg: raw-amount rank ≈ 9,500 → USD rank 4", fs=13, color=CRIM[6])
lab(1150, 514, 630, 24, "(6-decimal token amounts are numerically tiny)", fs=12, color="#777777")

# ======================================================== P3 protocols and account corpus
panel(1815, 15, 770, 560, "③ Frozen Protocols & Accounts", "#f2faf5", fs=26)
f.rect(1845, 74, 18, 18, "", NAVY[4], NAVY[6], arc=0, sw=0.6); lab(1866, 70, 50, 24, "train", fs=13, align="left")
f.rect(1925, 74, 18, 18, "", GOLD[3], GOLD[5], arc=0, sw=0.6); lab(1946, 70, 50, 24, "test", fs=13, align="left")
lab(2395, 70, 120, 24, "chance", fs=13, color=CRIM[6])
protos = ["random", "temporal", "LOBO→Ronin", "LOBO→Nomad"]
prev = ["0.003", "0.589", "1.5×10<sup>−5</sup>", "0.014"]
bx0, bw0 = 1990, 395
for k, (nm, pv) in enumerate(zip(protos, prev)):
    y = 112 + k * 52
    lab(1835, y, 150, 30, nm, fs=14, align="right")
    if k == 0:
        for j in range(20):
            f.rect(bx0 + j * bw0 / 20, y + 2, bw0 / 20, 26, "", GOLD[3] if j in (2, 7, 11, 16) else NAVY[4], "#ffffff", arc=0, sw=0.6)
    elif k == 1:
        f.rect(bx0, y + 2, 300, 26, "", NAVY[4], "#ffffff", arc=0, sw=0.6)
        f.rect(bx0 + 300, y + 2, bw0 - 300, 26, "Nomad", GOLD[3], "#ffffff", fs=12, arc=0, sw=0.6)
        f.arrow([(bx0 + 300, y - 4), (bx0 + 300, y + 34)], head=False, dashed=True, color=CRIM[6], sw=1.3)
    else:
        tr, te = ("Nomad", "Ronin") if k == 2 else ("Ronin", "Nomad")
        cut = 120 if k == 2 else 300
        f.rect(bx0, y + 2, cut, 26, tr, NAVY[4], "#ffffff", fs=12, arc=0, sw=0.6, color="#ffffff")
        f.rect(bx0 + cut, y + 2, bw0 - cut, 26, te, GOLD[3], "#ffffff", fs=12, arc=0, sw=0.6)
    lab(2395, y + 2, 140, 26, pv, fs=14, color=CRIM[6])
lab(1840, 322, 720, 28, "10 seeds · mean ± SD · 95 % CI · paired tests", fs=17, color=JADE[6], style="fontStyle=1;")
# account corpus
cyl(1845, 382, 92, 92, JADE[4], JADE[6], "Across", fs=14)
lab(1830, 478, 125, 40, "12.35 M cctx<br>9 bridges", fs=12, color="#555555")
f.rect(1958, 398, 200, 50, "fee + in − out < 0", "#ffffff", CRIM[5], fs=14, arc=20, sw=1.4)
lab(1958, 450, 200, 22, "1,284 violations", fs=12, color=CRIM[6])
f.arrow([(1939, 423), (1955, 423)], sw=1.2)
matrix(2180, 385, 6, 8, 14, NAVY, seed=2.3)
lab(2160, 472, 150, 22, "24 × 8 per account", fs=12, color="#555555")
f.arrow([(2160, 423), (2177, 423)], sw=1.2)
rows_list(2340, 380, 7, 150, 11, hot={0, 1, 3}, gap=3)
f.rect(2334, 375, 162, 46, "", "none", CRIM[5], arc=6, sw=1.4, extra="dashed=1;fillColor=none;")
lab(2500, 384, 80, 26, "top-<i>k</i>", fs=14, color=CRIM[6], style="align=left;")
f.arrow([(2294, 423), (2337, 423)], sw=1.2)
lab(2335, 496, 200, 22, "<i>k</i> ∈ {100, 250, 500}", fs=12, color="#555555")
lab(1830, 528, 740, 26, "190,337 accounts · 259 exploiters · test accounts unseen in training", fs=13, color="#555555")

# ======================================================== P4 BiMamba block zoom-in
panel(15, 590, 1685, 560, "④ Bidirectional Selective SSM Block (zoom-in)", "#f8f4fb", fs=26)
X0, Y0 = 45, 690
for r in range(8):
    strip(X0, Y0 + r * 24, 3, 24, PLUM, seed=r)
lab(30, 886, 110, 24, "<i>H</i> (<i>L</i>×<i>d</i>)", fs=15)
f.rect(118, 700, 12, 175, "", JADE[2], JADE[5], arc=30, sw=1)
lab(104, 676, 40, 22, "LN", fs=13, color=JADE[6])
f.arrow([(119 - 2, 787), (117, 787)], head=False)
f.arrow([(132, 787), (150, 787)], sw=1.1)
cube(152, 680, 46, 215, PLUM[3], PLUM[6], depth=10)
lab(140, 900, 70, 24, "<i>W</i><sub>in</sub>", fs=15)
lab(130, 922, 90, 20, "<i>d</i> → 2<i>d</i>", fs=12, color="#555555")
# x' and gate z
f.arrow([(202, 787), (238, 787)], sw=1.0)
strip(242, 775, 4, 24, NAVY, seed=0.2); lab(340, 773, 30, 26, "<i>x</i>′", fs=17)
f.arrow([(202, 700), (215, 672), (238, 672)], sw=1.0)
strip(242, 660, 4, 24, PLUM, seed=1.4); lab(340, 658, 30, 26, "<i>z</i>", fs=17)
# input-dependent Delta, B, C
for yy in (712, 787, 862):
    f.arrow([(338, 787), (402, yy)], sw=0.9)
strip(405, 700, 4, 24, GOLD, seed=0.5); lab(503, 698, 40, 26, "Δ<sub>t</sub>", fs=16)
strip(405, 775, 4, 24, JADE, seed=1.0); lab(503, 773, 40, 26, "<i>B</i><sub>t</sub>", fs=16)
strip(405, 850, 4, 24, JADE, seed=2.0); lab(503, 848, 40, 26, "<i>C</i><sub>t</sub>", fs=16)
curve_icon(556, 692, 46, 40, softplus, GOLD[1], GOLD[5], GOLD[6])
f.arrow([(535, 712), (554, 712)], sw=0.9)
f.rect(585, 748, 130, 30, "<i>A</i><0, <i>N</i>=16", "#ffffff", PLUM[5], fs=12, arc=20, sw=1.2)
op(660, 712, "e<sup>Δ<i>A</i></sup>", r=24, fs=14)
f.arrow([(604, 712), (634, 712)], sw=0.9); f.arrow([(660, 748), (660, 738)], sw=0.9)
# recurrence chain (forward scan)
lab(705, 700, 380, 24, "forward selective scan  →  time", fs=14, color=PLUM[6], style="fontStyle=2;")
hx = [735, 815, 895, 975, 1055]
for k, x in enumerate(hx):
    lb = ["<i>h</i><sub>1</sub>", "<i>h</i><sub>2</sub>", "<i>h</i><sub>3</sub>", "⋯", "<i>h</i><sub><i>L</i></sub>"][k]
    node(x, 800, 24, PLUM[1] if k != 3 else "#ffffff", PLUM[6] if k != 3 else "#ffffff", lb, 16)
    if k < 4:
        f.arrow([(x + 24, 800), (hx[k + 1] - 25, 800)], sw=1.3, color=PLUM[6])
f.arrow([(684, 712), (735, 712), (735, 774)], sw=0.9, color=GOLD[6])
f.arrow([(500, 800), (708, 800)], sw=0.9, color=JADE[6])
lab(560, 806, 140, 22, "Δ<sub>t</sub><i>B</i><sub>t</sub><i>x</i>′<sub>t</sub>", fs=13, color=JADE[6])
lab(700, 830, 400, 24, "<i>h</i><sub>t</sub> ∈ ℝ<sup><i>d</i>×<i>N</i></sup>, linear in <i>L</i>", fs=13, color="#555555")
# readout, skip, gate, out-projection
op(1130, 800, "⊗", fs=22)
f.arrow([(1079, 800), (1110, 800)], sw=1.1)
f.arrow([(500, 874), (500, 892), (1130, 892), (1130, 820)], sw=0.9, color=JADE[6])
op(1195, 800, "+", fs=22)
f.arrow([(1150, 800), (1175, 800)], sw=1.1)
f.rect(1160, 840, 72, 30, "<i>D</i>⊙<i>x</i>′", "#ffffff", NAVY[5], fs=13, arc=20, sw=1.1)
f.arrow([(1195, 840), (1195, 820)], sw=0.9)
curve_icon(1235, 650, 46, 40, silu, GOLD[1], GOLD[5], GOLD[6])
f.arrow([(338, 672), (1233, 672)], sw=0.9, color=PLUM[5])
op(1258, 800, "⊙", fs=22)
f.arrow([(1215, 800), (1238, 800)], sw=1.1); f.arrow([(1258, 692), (1258, 780)], sw=0.9, color=PLUM[5])
cube(1300, 705, 40, 190, PLUM[3], PLUM[6], depth=8)
f.arrow([(1278, 800), (1298, 800)], sw=1.1)
lab(1290, 900, 70, 22, "<i>W</i><sub>out</sub>", fs=14)
# backward branch
f.arrow([(124, 875), (124, 988), (238, 988)], sw=1.0)
f.rect(240, 960, 1100, 56, "⇄ flip  →  selective SSM, same form and separate weights  →  ⇄ flip", PLUM[1], PLUM[4], fs=16, arc=18, sw=1.3)
op(1405, 900, "‖", fs=22)
f.arrow([(1342, 800), (1405, 800), (1405, 880)], sw=1.1)
f.arrow([(1342, 988), (1405, 988), (1405, 920)], sw=1.1)
cube(1445, 840, 40, 120, PLUM[2], PLUM[6], depth=7)
f.arrow([(1425, 900), (1443, 900)], sw=1.1)
lab(1420, 962, 90, 20, "2<i>d</i> → <i>d</i>", fs=12, color="#555555")
op(1545, 900, "⊕", fs=22)
f.arrow([(1487, 900), (1525, 900)], sw=1.1)
f.arrow([(57, 882), (57, 1050), (1545, 1050), (1545, 920)], dashed=True, color="#555555", sw=1.1)
lab(1565, 845, 60, 30, "×2", fs=22)
f.arrow([(1565, 900), (1712, 900)], sw=1.4)
lab(40, 1072, 1300, 34, "<i>h</i><sub>t</sub> = e<sup>Δ<sub>t</sub><i>A</i></sup> ⊙ <i>h</i><sub>t−1</sub> + Δ<sub>t</sub><i>B</i><sub>t</sub><i>x</i>′<sub>t</sub>,   "
    "<i>y</i><sub>t</sub> = <i>C</i><sub>t</sub><i>h</i><sub>t</sub> + <i>D</i><i>x</i>′<sub>t</sub>,   "
    "out = <i>W</i><sub>out</sub>(<i>y</i> ⊙ SiLU(<i>z</i>))", fs=17, align="left")
lab(1350, 1072, 330, 34, "no attention · pure PyTorch", fs=14, color=PLUM[6], align="left")

# ======================================================== P5 pooling and KAN head
panel(1715, 590, 870, 560, "⑤ Pooling & KAN Head", "#fbf6ef", fs=26)
f.rect(1740, 820, 12, 160, "", JADE[2], JADE[5], arc=30, sw=1)
for r in range(6):
    strip(1765, 830 + r * 22, 3, 22, PLUM, seed=r + 2)
lab(1752, 966, 90, 22, "<i>H</i> (<i>L</i>×48)", fs=13)
f.arrow([(1833, 860), (1905, 806)], sw=0.9); f.arrow([(1833, 900), (1905, 868)], sw=0.9)
strip(1910, 794, 4, 22, PLUM, seed=0.3); lab(2003, 792, 70, 24, "mean", fs=13, align="left")
strip(1910, 856, 4, 22, PLUM, seed=1.3); lab(2003, 854, 70, 24, "last", fs=13, align="left")
strip(1910, 918, 4, 22, NAVY, seed=0.8); lab(2003, 916, 70, 24, "<b>c</b>", fs=14, align="left")
strip(1910, 980, 1, 22, GOLD, seed=0.8); lab(1937, 978, 120, 24, "<i>v</i> (triage)", fs=13, align="left")
op(2080, 900, "‖", fs=22)
for yy in (805, 867, 929, 991):
    f.arrow([(2050, yy), (2062, 896)], sw=0.8)
lab(2060, 926, 40, 22, "106", fs=13, color="#555555")
# KAN mini-network with spline edges
IN = [(2135, 790 + 52 * k) for k in range(6)]
HID = [(2275, 830 + 56 * k) for k in range(4)]
OUT = (2405, 914)
for a in IN:
    for b in HID:
        f.arrow([(a[0] + 9, a[1]), (b[0] - 11, b[1])], head=False, sw=0.6, color="#9a9a9a")
for b in HID:
    f.arrow([(b[0] + 11, b[1]), (OUT[0] - 13, OUT[1])], head=False, sw=0.8, color="#7a7a7a")
for k, a in enumerate(IN):
    node(*a, 9, [NAVY[3], PLUM[3], NAVY[3], PLUM[3], NAVY[4], GOLD[3]][k], "#555555")
for b in HID:
    node(*b, 11, GOLD[2], GOLD[6])
node(*OUT, 13, CRIM[2], CRIM[6])
f.arrow([(2100, 900), (2122, 900)], sw=1.0)
curve_icon(2170, 740, 52, 36, lambda t: math.tanh(t) + 0.3 * math.sin(2 * t), "#ffffff", PLUM[5], PLUM[6])
curve_icon(2180, 1005, 52, 36, lambda t: 0.5 * t * t - 0.08 * t ** 3, "#ffffff", PLUM[5], PLUM[6])
curve_icon(2318, 846, 52, 36, lambda t: 1 / (1 + math.exp(-1.6 * t)), "#ffffff", PLUM[5], PLUM[6])
op(2475, 914, "σ", fill=GOLD[2], fs=21)
f.arrow([(2420, 914), (2455, 914)], sw=1.1)
lab(2496, 900, 80, 30, "<i>p</i><sub>θ</sub>(<i>a</i>)", fs=16)
lab(2105, 1062, 470, 24, "106 → 24 → 1 · cubic B-splines, 5-interval grid", fs=13, color="#555555")
lab(2110, 1088, 460, 30, "φ(<i>x</i>) = <i>w</i>·SiLU(<i>x</i>) + Σ<sub><i>i</i></sub> <i>c</i><sub><i>i</i></sub><i>B</i><sub><i>i</i></sub>(<i>x</i>)", fs=16)
f.rect(1745, 655, 820, 46, "ℒ = BCE<sub><i>w</i>+</sub>(<i>y</i>, <i>p</i><sub>θ</sub>),  <i>w</i><sub>+</sub> ≤ 300 · AdamW 2×10<sup>−3</sup> · training only",
       "#ffffff", JADE[5], fs=15, arc=10, sw=1.3, extra="dashed=1;")

# ======================================================== P6 neuro-symbolic triage
panel(15, 1165, 1585, 520, "⑥ Neuro-Symbolic Alert Triage", "#fdf4f3", fs=26)
f.vertex("🔔", "text;html=1;fontSize=28;align=center;verticalAlign=middle;", 40, 1228, 44, 44)
lab(86, 1236, 200, 28, "𝒜 from ①", fs=17, align="left", style="fontStyle=1;")
rows_list(45, 1285, 11, 170, 16, hot={3, 8}, gap=5)
f.arrow([(217, 1320), (258, 1318)], sw=1.3); f.arrow([(217, 1460), (258, 1462)], sw=1.3)
f.rect(260, 1290, 290, 56, "<i>s</i><sub>M</sub>(<i>a</i>) = <i>p</i><sub>θ</sub>(<i>a</i>)  from ⑤", PLUM[1], PLUM[5], fs=16, arc=14, sw=1.4)
lab(258, 1352, 300, 22, "trained only on the other bridge's alerts", fs=11, color="#444444")
f.rect(260, 1434, 290, 56, "<i>s</i><sub>U</sub>(<i>a</i>) = log<sub>10</sub>(1 + <i>v</i>(<i>a</i>))", GOLD[1], GOLD[5], fs=16, arc=14, sw=1.4)
wlab(258, 1494, 300, 24, "untrained magnitude prior", fs=12)
# percentile bars
f.rect(600, 1262, 26, 112, "", PLUM[1], PLUM[5], arc=0, sw=1.0, grad=PLUM[5])
f.rect(600, 1406, 26, 112, "", GOLD[1], GOLD[5], arc=0, sw=1.0, grad=GOLD[5])
lab(630, 1300, 50, 26, "π<sub>M</sub>", fs=17, align="left"); lab(630, 1446, 50, 26, "π<sub>U</sub>", fs=17, align="left")
f.arrow([(552, 1318), (597, 1318)], sw=1.2); f.arrow([(552, 1462), (597, 1462)], sw=1.2)
lab(560, 1378, 120, 22, "percentile ranks", fs=11, color="#666666")
# fusion operators
op(790, 1318, "+", r=24, fs=24); lab(815, 1284, 40, 24, "α", fs=16, color=PLUM[6])
op(790, 1462, "max", r=27, fs=14)
for (ya, yb) in [(1318, 1318), (1318, 1452), (1462, 1328), (1462, 1462)]:
    f.arrow([(672, ya), (764, yb)], sw=0.9, color="#555555")
lab(828, 1326, 300, 26, "<i>h</i><sub>α</sub> = α π<sub>M</sub> + (1−α) π<sub>U</sub>", fs=14, align="left")
lab(828, 1470, 300, 26, "<i>h</i><sub>max</sub> = max(π<sub>M</sub>, π<sub>U</sub>)", fs=14, align="left")
f.rect(260, 1560, 560, 72, "", "#ffffff", JADE[5], arc=10, sw=1.3, extra="dashed=1;")
lab(270, 1563, 540, 24, "rank guarantee of the escalation rule", fs=13, color=JADE[6], style="fontStyle=2;")
lab(270, 1590, 540, 34, "rank<sub>max</sub>(<i>a</i>) ≤ 2 min{<i>r</i><sub>M</sub>(<i>a</i>), <i>r</i><sub>U</sub>(<i>a</i>)} − 1", fs=17)
# ranked queue for the analyst
f.arrow([(816, 1318), (1065, 1318), (1065, 1380), (1088, 1380)], sw=1.2)
f.arrow([(819, 1462), (1065, 1462), (1065, 1400), (1088, 1400)], sw=1.2)
lab(1088, 1236, 160, 28, "ranked queue", fs=17, align="left", style="fontStyle=1;")
rows_list(1092, 1280, 12, 130, 16, hot={0, 1, 2, 4}, gap=5)
f.rect(1085, 1274, 144, 88, "", "none", CRIM[5], arc=6, sw=1.5, extra="dashed=1;fillColor=none;")
f.vertex("👤", "text;html=1;fontSize=30;align=center;verticalAlign=middle;", 1240, 1282, 40, 40)
lab(1236, 1324, 60, 40, "top-<i>k</i><br>review", fs=13, color=CRIM[6])
# incident replay directions
cyl(1345, 1262, 100, 96, NAVY[5], NAVY[6], "Ronin<br>15,326", fs=13)
cyl(1475, 1262, 100, 96, PLUM[4], PLUM[6], "Nomad<br>1,070", fs=13)
f.arrow([(1395, 1362), (1395, 1405), (1525, 1405), (1525, 1362)], sw=1.3, color=NAVY[6])
f.arrow([(1545, 1258), (1545, 1236), (1375, 1236), (1375, 1258)], sw=1.3, color=PLUM[6])
lab(1312, 1412, 280, 22, "train on one bridge → rank the other", fs=12, color="#555555")
wlab(1315, 1442, 275, 110, "zero-day replay of two incidents; supervision = the other bridge's exploits only (2 or 463 positives)", fs=12)

# ======================================================== P7 federation and explanation
panel(1615, 1165, 970, 520, "⑦ Federation & Explanation", "#f3f6f9", fs=26)
f.rect(1640, 1228, 440, 440, "", "#ffffff", JADE[4], arc=6, sw=1.2, extra="dashed=1;")
lab(1650, 1236, 420, 24, "feasibility prototype · 10 seeds", fs=13, color=JADE[6], style="fontStyle=2;")
node(1760, 1310, 32, JADE[2], JADE[6], "Σ", 22)
lab(1800, 1282, 270, 22, "mean · median · 20 % trimmed", fs=12, color="#555555", style="align=left;")
lab(1800, 1306, 270, 22, "<i>T</i> = 8 rounds × <i>E</i> = 2", fs=12, color="#555555", style="align=left;")
clients = [(1690, "R-A", CRIM[3], CRIM[6]), (1800, "R-B", NAVY[2], NAVY[5]), (1920, "N-A", PLUM[2], PLUM[5]), (2030, "N-B", PLUM[2], PLUM[5])]
for x, nm, fl, st in clients:
    node(x, 1478, 26, fl, st, nm, 13)
    f.arrow([(x + (1760 - x) * 0.08, 1452), (1760 + (x - 1760) * 0.2, 1342)], sw=1.0, color="#555555")
lab(1648, 1508, 90, 22, "⚠ adversary", fs=12, color=CRIM[6])
lab(1645, 1540, 420, 22, "label flip · boosted model replacement (update × 4)", fs=12, color="#444444")
lab(1650, 1570, 420, 24, "clients = bridge × account half", fs=12, color="#555555")
lab(2110, 1232, 460, 26, "intrinsic: learned KAN edge functions", fs=14, color=PLUM[6], style="fontStyle=2;")
fns = [(lambda t: 1 / (1 + math.exp(-2 * (t - 0.5))), "w-ratio<sub>1h</sub>"),
       (lambda t: math.tanh(0.8 * t) + 0.1 * t, "new-account share"),
       (lambda t: -0.3 * t + 0.6 * math.exp(-t * t), "log Δ<i>t</i>")]
for k, (fn, nm) in enumerate(fns):
    curve_icon(2118 + k * 148, 1266, 130, 88, fn, "#ffffff", PLUM[4], PLUM[6])
    lab(2112 + k * 148, 1356, 142, 22, nm, fs=12)
lab(2110, 1404, 460, 26, "post hoc: permutation importance", fs=14, color=GOLD[6], style="fontStyle=2;")
imp = [("w-ratio<sub>1h</sub>", 0.95), ("new-acct", 0.72), ("amount", 0.55), ("Δ<i>t</i>", 0.42), ("count", 0.26)]
for k, (nm, v) in enumerate(imp):
    yy = 1440 + k * 32
    lab(2110, yy, 100, 26, nm, fs=12, align="right")
    f.rect(2218, yy + 4, 330 * v, 20, "", GOLD[6 - k // 2], GOLD[6], arc=0, sw=0.8)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "methodology.drawio")
f.save(out)

if "--export" in sys.argv:
    exe = r"C:\Program Files\draw.io\draw.io.exe"
    for fmt, extra in (("pdf", ["--crop"]), ("png", ["--scale", "2"])):
        dst = out.replace(".drawio", f".{fmt}")
        subprocess.run([exe, "--export", "--format", fmt, *extra, "--output", dst, out], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("exported", dst)
