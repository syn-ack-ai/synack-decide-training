"""Benchmark comparison card (PNG): SynACK Decide vs Jev-style models on the Decision Index, plus PR routing.

Our column comes from the 3,000-row sample scores (runs/eval-v3/lb-score-<ver>/scores.json) and the PR test
predictions (runs/eval-v3/results/pr_preds_<ver>.jsonl); the other columns from runs/board/index.json.

  uv run --with pillow --with matplotlib python runs/board_card.py [--ver v4] [--pr-threshold 0.38] [--out PATH]
"""
import argparse
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--ver", default="v4")
p.add_argument("--pr-threshold", type=float, default=0.38, help="v4 runs high: 0.38; a calibrated model: ~0.27")
p.add_argument("--out", default=None)
p.add_argument("--label", default=None, help="version shown on the card (default: --ver)")
p.add_argument("--scores", help="scores.json of a FULL Decision Index run (default: the 3,000-row sample scores)")
a = p.parse_args()
LABEL = a.label or a.ver

board = json.load(open(ROOT / "runs/board/index.json"))
ours = json.load(open(a.scores or ROOT / f"runs/eval-v3/lb-score-{a.ver}/scores.json"))
OFFICIAL = bool(a.scores)
by = {m["name"]: m for m in board["models"]}
others = [board["jev"], by["Surogate Rune 26B-A4B v3"], by["pplx-decider-v1-27b"], by["Winnow-12B"], by["Tev1-4B-experimental"]]
bench_id = {v["short"]: int(k) for k, v in board["benchmarks"].items()}


def index_score(m):
    return ours["decision_index"] if m is None else m["scores"]["balanced_skill"]


def cat_score(m, cid):
    if m is None:
        return next(x["skill"] for x in ours["areas"] if x["id"] == cid) * 100
    c = next((c for c in m["categories"] if c["id"] == cid), None)
    return c["skill"] * 100 if c and c.get("skill") is not None else None


def bench_score(m, short):
    r = (ours["benchmarks"] if m is None else m["results"]).get(str(bench_id[short]))
    return None if not r or r.get("score") is None or (m is None and not r.get("answered")) else r["score"] * 100


def pr_balanced_accuracy(pred_of, ids, gold):
    pos = [i for i in ids if gold[i]]
    neg = [i for i in ids if not gold[i]]
    return 50 * (sum(pred_of(i) for i in pos) / len(pos) + sum(not pred_of(i) for i in neg) / len(neg))


gold = {r["id"]: r["expected"]["needs_review"] == "yes" for r in map(json.loads, open(ROOT / "prcx/data/dataset/test_records.jsonl"))}
pr_p = {d["id"]: d["p_yes"] for d in map(json.loads, open(ROOT / f"runs/eval-v3/results/pr_preds_{a.ver}.jsonl"))}
script = {f"prcx:{r['repo']}#{r['number']}": float(r["score"])
          for r in csv.DictReader(open(ROOT / "prcx/data/dataset/features.csv")) if r["split"] == "test"}
pr_ids = [i for i in pr_p if i in gold]
pr_ours = pr_balanced_accuracy(lambda i: pr_p[i] >= a.pr_threshold, pr_ids, gold)
pr_script = pr_balanced_accuracy(lambda i: script[i] >= 0.30, [i for i in pr_ids if i in script], gold)
pr_pos_rate = 100 * sum(gold[i] for i in pr_ids) / len(pr_ids)
pr_repos = len({i.split("#")[0] for i in pr_ids})

cols = [None] + others  # None = ours
layout = [  # (group, benchmark row label, key): key is a category id, a board benchmark short name, or a special
    ("Overall", "Decision Index", "index"),
    ("Knowledge & Reasoning", "Category score", "knowledge"),
    ("", "GPQA Diamond", "GPQA Diamond"), ("", "GSM8K", "GSM8K"), ("", "CRUXEval", "CRUXEval"), ("", "MMLU-Pro", "MMLU-Pro"),
    ("Language Understanding", "Category score", "language"),
    ("", "ContractNLI", "ContractNLI"), ("", "ANLI", "ANLI"), ("", "HellaSwag", "HellaSwag"),
    ("Retrieval & Classification", "Category score", "retrieval"),
    ("", "BANKING77", "BANKING77"), ("", "CLINC150", "CLINC150"), ("", "HoVer", "HoVer"),
    ("Tools & Automation", "Category score", "tools"),
    ("", "BFCL", "BFCL"), ("", "API-Bank", "API-Bank"), ("", "When2Call", "When2Call"),
    ("Arts & Human Taste", "Category score", "arts"),
    ("", "Humicroedit", "Humicroedit"), ("", "New Yorker caption matching", "New Yorker"),
    ("PR Routing‡", "Balanced accuracy", "pr"),
    ("Speed", "Median latency (ms)†", "latency"),
]
rows = []
for group, label, key in layout:
    if key == "index":
        vals = [index_score(m) for m in cols]
    elif key == "pr":
        vals = [pr_ours] + [None] * len(others)
    elif key == "latency":
        vals = [ours["latency_ms"]["median"]] + [m["latency"].get("median") for m in others]
    elif key in bench_id:
        vals = [bench_score(m, key) for m in cols]
    else:
        vals = [cat_score(m, key) for m in cols]
    rows.append((group, label, *[round(v, 1) if isinstance(v, float) else v for v in vals]))
rows.append(("Size", "Parameters (active)", "26B (4B)", "n/a", "26B (4B)", "28B", "12B", "4.7B"))

for r in rows:
    print(" | ".join("—" if v is None else str(v) for v in r))

model_names = [
    f"SynACK Decide\n26B-A4B {LABEL}" + ("" if OFFICIAL else "*"),
    "Jev\n(TypeSafe)",
    "Surogate Rune\n26B-A4B v3",
    "pplx-decider-\nv1-27b",
    "Winnow-12B",
    "Tev1-4B\n(Together AI)",
]


# Winner rule: exactly ONE winner per scored row, and only when at least two models have a value.
# Higher is better for scores; lower is better for latency.
# If there is an exact tie, choose the left-most model so only one cell is highlighted.
def winner_index(bench, vals):
    if bench == "Parameters (active)":
        return None
    numeric = [(i, v) for i, v in enumerate(vals) if isinstance(v, (int, float))]
    if len(numeric) < 2:
        return None
    if bench.startswith("Median latency"):
        best = min(v for _, v in numeric)
    else:
        best = max(v for _, v in numeric)
    return next(i for i, v in numeric if v == best)


footnotes = [
    (f"SynACK Decide {LABEL}: complete Decision Index 0.2.1 run (155,390 rows), scored with the leaderboard's own scorer; latency is our median on one RTX PRO 6000." if OFFICIAL else
     f"* SynACK Decide {LABEL} is an estimate from a stratified 3,000-row sample of the Decision Index suite; overall uncertainty is roughly ±3–4 points."),
    "Other model columns are full-suite leaderboard results dated 2026-09-28. † Jev latency is hosted API round-trip; other latency values are single-GPU.",
    f"‡ PR routing (not part of the Decision Index): will a PR need substantive changes in review? {len(pr_ids):,} held-out PRs from {pr_repos} unseen public repos "
    f"({pr_pos_rate:.0f}% do).",
    f"   Balanced accuracy at p ≥ {a.pr_threshold:.2f} (always answering \"no\" scores 50%); the complexity-scorer baseline scores {pr_script:.1f}%. "
    "Other models were not evaluated on PRs.",
]

# Canvas and fonts
W = 2500
row_h = 58
H = 235 + 112 + row_h * len(rows) + 28 + 44 + 26 * len(footnotes) + 40
img = Image.new("RGB", (W, H), (6, 17, 31))
draw = ImageDraw.Draw(img)


def font_dir():
    try:
        import matplotlib
        d = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf"
        if (d / "DejaVuSans.ttf").exists():
            return d
    except ImportError:
        pass
    return Path("/usr/share/fonts/truetype/dejavu")


FONT_DIR = font_dir()


def font(name, size):
    return ImageFont.truetype(str(FONT_DIR / name), size)


title_font = font("DejaVuSans-Bold.ttf", 76)
subtitle_font = font("DejaVuSans.ttf", 32)
header_font = font("DejaVuSans-Bold.ttf", 24)
cell_font = font("DejaVuSans.ttf", 22)
cell_bold = font("DejaVuSans-Bold.ttf", 22)
small_font = font("DejaVuSans.ttf", 17)
foot_font = font("DejaVuSans.ttf", 15)

# Palette
BG = (6, 17, 31)
PANEL = (10, 26, 44)
PANEL_ALT = (13, 34, 57)
LINE = (48, 86, 118)
TEXT = (236, 244, 255)
MUTED = (165, 190, 214)
SYN_BLUE = (24, 78, 132)
SYN_BORDER = (96, 184, 255)
WINNER_FILL = (18, 113, 63)
WINNER_BORDER = (91, 235, 132)
WINNER_TEXT = (206, 255, 216)
GROUP_FILL = (8, 24, 40)
CATEGORY_FILL = (14, 39, 65)

# Accent
draw.ellipse((1600, -540, 2590, 450), outline=(22, 85, 144), width=3)
draw.ellipse((1680, -470, 2510, 360), outline=(16, 55, 98), width=2)

# Title
draw.text((60, 42), f"SynACK Decide 26B-A4B {LABEL}", font=title_font, fill=TEXT)
draw.text((64, 130), "Decision Index benchmark comparison", font=subtitle_font, fill=(195, 217, 239))
draw.text((64, 174), "One winner per row • highest score wins • latency uses lowest value", font=small_font, fill=(108, 188, 255))

# Table geometry
x0, y0 = 50, 235
group_w, bench_w = 330, 430
model_ws = [310, 220, 285, 265, 210, 270]
header_h = 112

col_starts = [x0, x0 + group_w, x0 + group_w + bench_w]
for w in model_ws[:-1]:
    col_starts.append(col_starts[-1] + w)
table_right = col_starts[-1] + model_ws[-1]

# Header
draw.rounded_rectangle((x0, y0, table_right, y0 + header_h), radius=16, fill=PANEL_ALT, outline=LINE, width=2)
draw.text((x0 + 18, y0 + 40), "Category", font=header_font, fill=TEXT)
draw.text((x0 + group_w + 18, y0 + 40), "Benchmark", font=header_font, fill=TEXT)

for i, hdr in enumerate(model_names):
    cx, cw = col_starts[2 + i], model_ws[i]
    if i == 0:
        draw.rounded_rectangle((cx, y0 - 8, cx + cw, y0 + header_h + 8),
                               radius=14, fill=(16, 49, 83), outline=SYN_BORDER, width=4)
    parts = hdr.split("\n")
    yy = y0 + (header_h - len(parts) * 27) // 2
    for part in parts:
        bbox = draw.textbbox((0, 0), part, font=header_font)
        draw.text((cx + (cw - (bbox[2] - bbox[0])) / 2, yy), part, font=header_font, fill=TEXT)
        yy += 28

# Row rendering and group spans
current_y = y0 + header_h
group_ranges = []

for row in rows:
    group, bench, *vals = row

    if group:
        if group_ranges:
            group_ranges[-1]["end"] = current_y
        group_ranges.append({"name": group, "start": current_y, "end": None})

    is_category = bench == "Category score"
    row_fill = CATEGORY_FILL if is_category else PANEL
    draw.rectangle((x0 + group_w, current_y, table_right, current_y + row_h), fill=row_fill)
    draw.text((x0 + group_w + 18, current_y + 16), bench,
              font=cell_bold if is_category else cell_font, fill=TEXT)

    win_idx = winner_index(bench, vals)

    for i, v in enumerate(vals):
        cx, cw = col_starts[2 + i], model_ws[i]

        # SynACK's column stays blue, but only the actual winner gets the green badge.
        if i == 0:
            draw.rectangle((cx, current_y, cx + cw, current_y + row_h), fill=SYN_BLUE)

        if win_idx is not None and i == win_idx:
            draw.rounded_rectangle((cx + 5, current_y + 5, cx + cw - 5, current_y + row_h - 5),
                                   radius=8, fill=WINNER_FILL, outline=WINNER_BORDER, width=2)

        if v is None:
            label = "—"
        elif bench.startswith("Median latency") and isinstance(v, (int, float)):
            label = f"{v:.0f}"
        elif isinstance(v, float):
            label = f"{v:.1f}%"
        else:
            label = str(v)

        use_font = cell_bold if (i == win_idx or i == 0) else cell_font
        use_fill = WINNER_TEXT if i == win_idx else TEXT
        bbox = draw.textbbox((0, 0), label, font=use_font)
        tw = bbox[2] - bbox[0]
        draw.text((cx + (cw - tw) / 2, current_y + 15), label, font=use_font, fill=use_fill)

    draw.line((x0 + group_w, current_y + row_h, table_right, current_y + row_h), fill=LINE, width=1)
    current_y += row_h

if group_ranges:
    group_ranges[-1]["end"] = current_y

# Group labels
for gr in group_ranges:
    sy, ey = gr["start"], gr["end"]
    draw.rectangle((x0, sy, x0 + group_w, ey), fill=GROUP_FILL, outline=LINE, width=1)
    words = gr["name"].split()
    lines, acc = [], ""
    for word in words:
        trial = (acc + " " + word).strip()
        if draw.textbbox((0, 0), trial, font=header_font)[2] > group_w - 40 and acc:
            lines.append(acc)
            acc = word
        else:
            acc = trial
    if acc:
        lines.append(acc)
    yy = sy + (ey - sy - len(lines) * 28) // 2
    for line_txt in lines:
        draw.text((x0 + 20, yy), line_txt, font=header_font, fill=(202, 223, 244))
        yy += 28

# Grid
for x in [x0 + group_w, x0 + group_w + bench_w] + col_starts[3:]:
    draw.line((x, y0, x, current_y), fill=LINE, width=1)

# SynACK outline
syn_x, syn_w = col_starts[2], model_ws[0]
draw.rounded_rectangle((syn_x, y0 - 8, syn_x + syn_w, current_y + 8),
                       radius=14, outline=SYN_BORDER, width=4)

# Legend
fy = current_y + 28
draw.rectangle((65, fy, 92, fy + 20), fill=SYN_BLUE, outline=SYN_BORDER, width=2)
draw.text((105, fy - 2), "SynACK column", font=small_font, fill=MUTED)
draw.rounded_rectangle((330, fy, 360, fy + 20), radius=5, fill=WINNER_FILL, outline=WINNER_BORDER, width=2)
draw.text((375, fy - 2), "Best value in row — exactly one winner", font=small_font, fill=MUTED)
draw.text((900, fy - 2), "Ties: left-most model selected", font=small_font, fill=MUTED)

# Footnotes
foot_y = fy + 44
for k, note in enumerate(footnotes):
    draw.text((65, foot_y + 26 * k), note, font=foot_font, fill=(142, 164, 187))

out = Path(a.out or ROOT / f"runs/board/synack_decide_{LABEL}_benchmark_card.png")
img.save(out, quality=96)
print(out)
