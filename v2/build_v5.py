"""v5 training mix: close the measured Decision Index gaps without giving up what v4 already wins.

New data (train splits or code generators only; every record goes through decontam.py before use):
  board/home_appliance  the harness generator (decision_index/suite/build/home_appliance.py) with offsets >= 10
                        (the suite uses offsets 0-9) plus its official dev split (households 0-7)
  board/pop909_synth    code-generated chord recognition in the POP909-CL format (same 129-option vocabulary);
                        progressions, voicings, arpeggios, bass and melody are rendered here, labels are exact
  board/isarcasm        iSarcasmEval official train split (task A, English), sarcastic class oversampled 2x
  board/vast            VAST train split (stance toward topic)
  board/habermas        Habermas Machine TRAIN rounds (the suite uses EVAL cohorts' IID/OOD test rounds)
  board/cfcolor         cfcolor train_vec ratings only (the suite's targets come from test_vec)
  board/winogrande      2,500 more WinoGrande train rows
  pr/injected_natural   PR description claims at the real 28% needs-review rate (v4 used 50/50 and drifted high)
  fin/tfns, fin/fpb     financial news / headline sentiment (Twitter financial news MIT; PhraseBank train split)
  fin/smartbugs         SmartBugs curated smart-contract vulnerability class (annotation comments stripped)
Replay: v4's training mix (old 50/50 injected PRs dropped). The HF job overwrites the replay rows' teacher with
v4's own probabilities (self-distillation), which anchors the benchmarks v4 already leads.

  python v2/build_v5.py            -> v2/data/v5/{new_records.jsonl, train.jsonl, valid.jsonl}
  python v2/build_v5.py --keep v2/data/v5/new_records.clean.jsonl   (after decontam.py)
"""
import argparse
import collections
import csv
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "v2/data/v5/raw"
sys.path.insert(0, str(ROOT / "v2"))
sys.path.insert(0, str(ROOT / "decision-index"))
from common import expand  # noqa: E402
from decision_index.suite.build import home_appliance as HA  # noqa: E402
from decision_index.suite.build.adapters_creative import ROOTS, TEMPLATES, VOCAB  # noqa: E402

csv.field_size_limit(10**9)


def rec(src, rid, state, qkey, q, gold):
    return {"id": f"{src}:{rid}", "source": src, "state": state, "questions": {qkey: q}, "expected": {qkey: gold}}


# ---------- Home appliance: harness generator, unseen offsets + dev split ----------
def home_appliance(rng, offsets=range(10, 18), per_row_targets=4):
    out = []
    rows = [HA.make_row(h, o) for h in range(len(HA.HOUSEHOLDS)) for o in offsets]
    rows += [HA.make_row(h, o) for h in range(8) for o in range(10)]  # official dev split
    for r in rows:
        targets_yes, targets_no, other = [], [], []
        for qid, q in r["questions"].items():
            role = r["metadata"]["roles"][qid] if "roles" in r["metadata"] else None
            text = q["instructions"]
            if text.startswith("Is ") and "justified device target set" in text:
                gold_text = q["criteria"][r["expected"][qid]]
                (targets_yes if gold_text.startswith("yes") else targets_no).append(qid)
            else:
                other.append(qid)
        keep = other + rng.sample(targets_yes, min(len(targets_yes), per_row_targets // 2)) + \
            rng.sample(targets_no, min(len(targets_no), per_row_targets - min(len(targets_yes), per_row_targets // 2)))
        for qid in keep:
            out.append({"id": f"board/home_appliance:{r['id']}:{qid}", "source": "board/home_appliance", "state": r["state"],
                        "questions": {qid: r["questions"][qid]}, "expected": {qid: r["expected"][qid]}})
    return out


# ---------- POP909-style chord recognition, generated ----------
CRITERIA = None


def pop909_criteria():
    global CRITERIA
    if CRITERIA is None:
        labels = {pc: "chord_" + str(i) for i, pc in enumerate(VOCAB)}
        CRITERIA = (labels, {labels[pc]: ("No chord sounding at the target beat." if pc == "NoChord" else
                                          "A chord pitch-class set outside this vocabulary." if pc == "Other" else
                                          {"names": VOCAB[pc], "pitch_classes": list(pc)}) for pc in VOCAB})
    return CRITERIA


MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]
TYPE_W = {"M": 30, "m": 22, "7": 9, "maj7": 7, "m7": 9, "sus4": 4, "sus2": 3, "dim": 2, "aug": 1, "half-dim7": 2,
          "dim7": 1, "mMaj7": 0.5, "aug7": 0.5}


def diatonic_chord(rng, key, minor):
    scale = MINOR if minor else MAJOR
    deg = rng.choices(range(7), weights=[5, 2, 2, 4, 5, 4, 1])[0]
    root = (key + scale[deg]) % 12
    third = (scale[(deg + 2) % 7] - scale[deg]) % 12
    fifth = (scale[(deg + 4) % 7] - scale[deg]) % 12
    base = "m" if third == 3 and fifth == 7 else "dim" if third == 3 and fifth == 6 else "M"
    r = rng.random()
    if r < 0.25:
        seventh = (scale[(deg + 6) % 7] - scale[deg]) % 12
        name = {("M", 11): "maj7", ("M", 10): "7", ("m", 10): "m7", ("dim", 10): "half-dim7", ("m", 11): "mMaj7"}.get((base, seventh), base)
    elif r < 0.33 and base == "M":
        name = rng.choice(["sus4", "sus2"])
    elif r < 0.38:  # chromatic / borrowed chord
        name = rng.choices(list(TYPE_W), weights=list(TYPE_W.values()))[0]
        root = rng.randrange(12)
    else:
        name = base
    return root, name


def pop909_synth(rng, n):
    labels, criteria = pop909_criteria()
    instr = ("Infer the chord sounding at the target beat from the score notes and musical context. Choose its pitch-class "
             "set; MIDI pitch classes are C=0 through B=11. Equivalent chord names are grouped.")
    out = []
    for i in range(n):
        key, minor = rng.randrange(12), rng.random() < 0.35
        bars = 5
        start_bar = rng.randint(4, 120)
        chords = []  # (start_beat, end_beat, root, name or None, extra pcs)
        b = start_bar * 4.0
        while b < (start_bar + bars) * 4:
            dur = rng.choice([4, 4, 4, 2, 8])
            root, name = diatonic_chord(rng, key, minor)
            extra = []
            r = rng.random()
            if r < 0.03:
                name = None  # NoChord
            elif r < 0.06:
                extra = [(root + rng.choice([2, 5, 9])) % 12]  # add9/add11/6 -> Other
            chords.append((b, b + dur, root, name, extra))
            b += dur
        target = start_bar * 4 + 2 + 4 * rng.randint(1, bars - 2)
        jit = (lambda: 0.0) if rng.random() < 0.5 else (lambda: rng.choice([0, 0, 1 / 48, -1 / 48, 1 / 24, 1 / 96]))
        notes = []
        style = rng.choice(["block", "arp", "alberti", "pulse"])
        melody_oct = rng.choice([5, 6]) * 12
        for cs, ce, root, name, extra in chords:
            if ce <= target - 4 or cs >= target + 4:
                continue
            if name is None:
                if rng.random() < 0.5:  # a lone melody fragment over no chord
                    p = melody_oct + (key + rng.choice(MAJOR)) % 12
                    notes.append((p, cs + 1, cs + 2))
                continue
            pcs = sorted({(root + x) % 12 for x in TEMPLATES[name]} | set(extra))
            bass = 36 + root + (0 if root < 8 else -12)
            notes.append((bass, cs, ce - rng.choice([0, 0.5, 1])))
            if rng.random() < 0.3:
                notes.append((bass + 12, cs + 2, ce))
            voicing = [48 + ((root + x) % 12) + (12 if (root + x) % 12 < 5 else 0) for x in TEMPLATES[name]] + [60 + e for e in extra]
            if len(voicing) >= 4 and rng.random() < 0.3:
                voicing.pop(2)  # omit the fifth sometimes
            inv = rng.randrange(len(voicing))
            voicing = sorted(voicing[inv:] + [v + 12 for v in voicing[:inv]])
            t = cs
            while t < ce:
                if style == "block":
                    d = rng.choice([1, 1, 2])
                    for v in voicing + ([voicing[-1] + 12] if rng.random() < 0.4 else []):
                        notes.append((v, t, min(ce, t + d)))
                    t += d
                elif style == "arp":
                    for k, v in enumerate(voicing + voicing[::-1][1:-1]):
                        if t + 0.5 * k >= ce:
                            break
                        notes.append((v, t + 0.5 * k, t + 0.5 * k + 0.5))
                    t += 0.5 * (2 * len(voicing) - 2)
                elif style == "alberti" and len(voicing) >= 3:
                    for k, v in enumerate([voicing[0], voicing[-1], voicing[1], voicing[-1]]):
                        notes.append((v, t + 0.5 * k, t + 0.5 * k + 0.5))
                    t += 2
                else:
                    for v in voicing:
                        notes.append((v, t, t + 0.5))
                    t += 1
            mt = cs
            scale = MINOR if minor else MAJOR
            while mt < ce:
                d = rng.choice([0.25, 0.5, 0.5, 0.5, 1])
                strong = abs(mt - round(mt)) < 1e-9 and int(mt) % 2 == 0
                pc = rng.choice(pcs) if strong or rng.random() < 0.5 else (key + rng.choice(scale)) % 12
                if rng.random() > 0.15:
                    notes.append((melody_oct + pc, mt, min(ce, mt + d)))
                mt += d
        window = [(p, a + jit(), b2 + jit()) for p, a, b2 in notes if a < target + 4 and b2 > target - 4]
        window.sort(key=lambda x: (x[2], x[0]))  # POP909 rows list notes in note-off order
        state = {"target_time_beats": target,
                 "context_notes": [{"pitch": p, "pitch_class": p % 12, "start_beat": a, "end_beat": b2} for p, a, b2 in window],
                 "public_context": {"ticks_per_beat": 480, "key_signatures": [], "time_signatures": []}}
        cur = next(c for c in chords if c[0] <= target < c[1])
        if cur[3] is None:
            gold = labels["NoChord"]
        else:
            pcs = tuple(sorted({(cur[2] + x) % 12 for x in TEMPLATES[cur[3]]} | set(cur[4])))
            gold = labels[pcs] if pcs in VOCAB else labels["Other"]
        out.append(rec("board/pop909_synth", i, state, "chord", {"type": "choice", "instructions": instr, "criteria": criteria}, gold))
    return out


# ---------- text benchmarks from their train splits ----------
def isarcasm(rng):
    out = []
    for i, row in enumerate(csv.DictReader(open(RAW / "isarcasm_train.En.csv", encoding="utf-8-sig"))):
        text = (row.get("tweet") or "").strip()
        if not text or row["sarcastic"] not in ("0", "1"):
            continue
        gold = "yes" if row["sarcastic"] == "1" else "no"
        q = {"type": "choice", "instructions": "Is this text intended to be sarcastic?", "criteria": {"no": "No", "yes": "Yes"}}
        for c in range(2 if gold == "yes" else 1):  # sarcastic class is the scored class and the minority
            out.append(rec("board/isarcasm", f"{i}.{c}", text, "sarcastic", q, gold))
    return out


def vast(rng, n):
    labels = ["against", "favor", "neutral"]
    rows = list(csv.DictReader(open(RAW / "vast_train.csv", encoding="utf-8")))
    seen, out = set(), []
    rng.shuffle(rows)
    for r in rows:
        k = (r["post"][:200], r["topic_str"])
        if k in seen:
            continue
        seen.add(k)
        prompt = f"Topic: {r['topic_str']}\nPost: {r['post']}\nDetermine the stance of the post toward the topic."
        out.append(rec("board/vast", r["new_id"], {}, "q1", {"type": "choice", "instructions": prompt,
                                                             "criteria": {"A": labels[0], "B": labels[1], "C": labels[2]}},
                       "ABC"[int(r["label"])]))
        if len(out) >= n:
            break
    return out


def winogrande(rng, n):
    from datasets import load_dataset
    ds = load_dataset("allenai/winogrande", "winogrande_xl", split="train")
    idx = rng.sample(range(len(ds)), n)
    out = []
    for i in idx:
        r = ds[i]
        out.append(rec("board/winogrande", i, {}, "q1", {"type": "choice", "instructions": "Which option correctly fills the blank?\n" + r["sentence"],
                                                        "criteria": {"A": r["option1"], "B": r["option2"]}}, "AB"[int(r["answer"]) - 1]))
    return out


def habermas(rng, n):
    import pyarrow.parquet as pq
    cols = ["metadata.version", "metadata.status", "round_id", "iteration_index", "question.split", "question.text",
            "rankings.metadata.status", "rankings.candidate_ids", "rankings.numerical_ranks", "candidates.metadata.id",
            "candidates.text", "own_opinion.metadata.id", "own_opinion.text", "other_opinions.metadata.id"]
    raw = pq.read_table(RAW / "hm_all_candidate_comparisons.parquet", columns=cols).to_pylist()
    groups = collections.defaultdict(list)
    for row in raw:
        if row["metadata.version"].startswith("EVAL") and row["question.split"] in {"IID_TEST", "OOD_TEST"}:
            continue  # the suite's cohorts
        if row["question.split"] != "TRAIN" or row["metadata.status"] != "COMPLETED" or row["rankings.metadata.status"] != "COMPLETED":
            continue
        if row["iteration_index"] != 0 or not row["rankings.candidate_ids"]:
            continue
        groups[(row["round_id"], tuple(sorted(row["rankings.candidate_ids"])))].append(row)
    out = []
    instr = ("Choose the consensus statement you predict this group would rank highest on average, given their expressed opinions. "
             "Each participant has equal weight. Assess the group's preferences, rather than your own policy preference.")
    for (round_id, cands), panel in sorted(groups.items()):
        ref = panel[0]
        expected_panel = set(ref["other_opinions.metadata.id"] or []) | {ref["own_opinion.metadata.id"]}
        if {r["own_opinion.metadata.id"] for r in panel} != expected_panel or not 2 <= len(cands) <= 255:
            continue
        text_map = dict(zip(ref["candidates.metadata.id"], ref["candidates.text"]))
        if not all(isinstance(text_map.get(c), str) and text_map[c].strip() for c in cands):
            continue
        sums = dict.fromkeys(cands, 0)
        ok = True
        for r in panel:
            ranks = dict(zip(r["rankings.candidate_ids"], r["rankings.numerical_ranks"]))
            if set(ranks) != set(cands):
                ok = False
                break
            for c, v in ranks.items():
                sums[c] += v
        best = min(sums.values())
        winners = [c for c in cands if sums[c] == best]
        if not ok or len(winners) != 1:
            continue
        order = sorted(cands, key=lambda c: hashlib.sha256((round_id + c).encode()).digest())
        keys = {c: f"option_{i}" for i, c in enumerate(order)}
        opinions = {r["own_opinion.metadata.id"]: r["own_opinion.text"] for r in panel}
        state = {"policy_question": ref["question.text"], "participant_opinions": [opinions[k] for k in sorted(opinions)]}
        out.append(rec("board/habermas", round_id, state, "consensus",
                       {"type": "choice", "instructions": instr, "criteria": {keys[c]: text_map[c] for c in order}}, keys[winners[0]]))
    rng.shuffle(out)
    return out[:n]


def cfcolor(rng, n):
    import numpy as np
    from scipy.io import loadmat
    base = RAW / "cfcolor/release"
    d = loadmat(base / "allMTurkRatings.mat", squeeze_me=True)
    t = loadmat(base / "themeData.mat", squeeze_me=True, struct_as_record=False)["datapoints"]
    rgb = np.asarray(t.rgb)
    hexs = lambda v: ["#" + "".join(f"{int(round(float(x) * 255)):02x}" for x in v[i:i + 3]) for i in range(0, 15, 3)]  # noqa: E731
    train = np.asarray(d["train_vec"])
    out = []
    users = sorted(set(train[:, 0].tolist()))
    rng.shuffle(users)
    for user in users:
        rs = collections.defaultdict(set)
        for _, p, r in train[train[:, 0] == user]:
            rs[int(p)].add(int(r))
        uniq = [(p, next(iter(v))) for p, v in rs.items() if len(v) == 1]
        if len(uniq) < 8:
            continue
        for _ in range(3):  # up to three target pairs per user, history from the user's other train ratings
            rng.shuffle(uniq)
            (pa, ra), (pb, rb) = uniq[0], uniq[1]
            if abs(ra - rb) < 2:
                continue
            hist = [{"colors": hexs(rgb[p - 1]), "rating": r} for p, r in uniq[2:2 + rng.randint(4, 8)]]
            a, b = ((pa, ra), (pb, rb)) if rng.random() < 0.5 else ((pb, rb), (pa, ra))
            state = {"history": hist, "rating_scale": "1 = lowest preference; 5 = highest preference"}
            q = {"type": "choice", "instructions": "Which five-color palette would this user rate more highly, given their earlier ratings?",
                 "criteria": {"A": hexs(rgb[a[0] - 1]), "B": hexs(rgb[b[0] - 1])}}
            out.append(rec("board/cfcolor", f"{user}:{a[0]}:{b[0]}", state, "preference", q, "A" if a[1] > b[1] else "B"))
        if len(out) >= n:
            break
    return out


# ---------- finance pack ----------
def finance(rng, n_tfns, n_fpb):
    from datasets import load_dataset
    out = []
    tf = load_dataset("zeroshot/twitter-financial-news-sentiment", split="train")
    names = ["bearish", "bullish", "neutral"]
    for i in rng.sample(range(len(tf)), n_tfns):
        r = tf[i]
        out.append(rec("fin/tfns", i, {"post": r["text"]}, "sentiment",
                       {"type": "choice", "instructions": "What market sentiment does this financial news post express?",
                        "criteria": {"bearish": "Bearish: negative for the company or market", "bullish": "Bullish: positive for the company or market",
                                     "neutral": "Neutral or no clear direction"}}, names[r["label"]]))
    fp = load_dataset("atrost/financial_phrasebank", split="train")
    names = ["negative", "neutral", "positive"]
    for i in rng.sample(range(len(fp)), min(n_fpb, len(fp))):
        r = fp[i]
        out.append(rec("fin/fpb", i, {"sentence": r["sentence"]}, "sentiment",
                       {"type": "choice", "instructions": "From an investor's point of view, is this financial news sentence positive, negative or neutral?",
                        "criteria": {"negative": None, "neutral": None, "positive": None}}, names[r["label"]]))
    cats = {"access_control": "Access control: missing or wrong authorization checks", "arithmetic": "Arithmetic: integer overflow or underflow",
            "bad_randomness": "Bad randomness: predictable on-chain randomness", "denial_of_service": "Denial of service: the contract can be blocked or made unusable",
            "front_running": "Front running: transaction ordering can be exploited", "reentrancy": "Reentrancy: external call before state is updated",
            "short_addresses": "Short address: malformed input lengths are not checked", "time_manipulation": "Time manipulation: logic depends on miner-controlled timestamps",
            "unchecked_low_level_calls": "Unchecked low-level calls: call/send return values are ignored", "other": "Other vulnerability"}
    for p in sorted((RAW / "smartbugs-curated/dataset").glob("*/*.sol")):
        src = p.read_text(errors="ignore")
        src = re.sub(r"/\*.*?\*/", "", src, count=1, flags=re.S)  # header block: @source / @author / @vulnerable_at_lines
        src = "\n".join(l for l in src.splitlines() if "<yes>" not in l and "<report>" not in l and "@vulnerable" not in l)
        src = re.sub(r"\n{3,}", "\n\n", src).strip()[:6000]
        out.append(rec("fin/smartbugs", p.parent.name + "/" + p.stem, {"contract": src}, "vulnerability",
                       {"type": "choice", "instructions": "Which vulnerability class does this Solidity contract contain?", "criteria": cats}, p.parent.name))
    return out


# ---------- PR claims at the natural rate ----------
def pr_injected(rng, n):
    sys.argv = sys.argv[:1]
    import build_v4
    prs = [json.loads(l) for l in open(ROOT / "prcx/data/dataset/train_records.jsonl")]
    yes = [r for r in prs if r["expected"]["needs_review"] == "yes"]
    no = [r for r in prs if r["expected"]["needs_review"] == "no"]
    ny = round(n * len(yes) / len(prs))
    out = [build_v4.inject(rng, r, 5) for r in rng.sample(yes, ny)] + [build_v4.inject(rng, r, 5) for r in rng.sample(no, n - ny)]
    for r in out:
        r["source"] = "pr/injected_natural"
    return out


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "v2/data/v5"))
    p.add_argument("--seed", type=int, default=20261005)
    p.add_argument("--keep", help="decontam.py output for new_records.jsonl")
    p.add_argument("--replay", type=int, default=17000)
    a = p.parse_args()
    out = Path(a.out)
    rng = random.Random(a.seed)
    new = (home_appliance(rng) + pop909_synth(rng, 1100) + isarcasm(rng) + vast(rng, 2500) + habermas(rng, 900) +
           cfcolor(rng, 1200) + winogrande(rng, 2500) + finance(rng, 1200, 1200) + pr_injected(rng, 2000))
    write(out / "new_records.jsonl", new)
    keep = {json.loads(l)["id"] for l in open(a.keep)} if a.keep else None
    def official_dev(r):  # the harness's own dev split (households 0-7, offsets 0-9): designated for development
        parts = r["id"].split(":")
        return r["source"] == "board/home_appliance" and parts[2] == "dev" and int(parts[4]) < 10
    new = [r for r in new if keep is None or r["id"] in keep or official_dev(r)]
    # hold out a slice of each new family for validation (records, never trained on)
    by = collections.defaultdict(list)
    for r in new:
        by[r["source"]].append(r)
    train_new, valid_new = [], []
    for src, rs in by.items():
        rng.shuffle(rs)
        k = min(80, len(rs) // 10)
        valid_new += rs[:k]
        train_new += rs[k:]
    ex_new = [e for r in train_new for e in expand(r, copies=1)]
    v4 = [json.loads(l) for l in open(ROOT / "v2/data/v4/train.jsonl")]
    v4 = [e for e in v4 if e["source"] != "prcx/injected"]
    rng.shuffle(v4)
    replay = v4[:a.replay]
    for e in replay:
        e["replay"] = True  # the job replaces these teachers with v4's own probabilities
    train = replay + ex_new
    rng.shuffle(train)
    write(out / "train.jsonl", train)
    valid = [json.loads(l) for l in open(ROOT / "v2/data/v4/valid.jsonl")] + [e for r in valid_new for e in expand(r, copies=1)]
    write(out / "valid.jsonl", valid)
    c = collections.Counter(e["source"].split(":")[0] if "/" in e["source"] else e["source"] for e in ex_new)
    print(f"train {len(train)} examples ({len(replay)} replay + {len(ex_new)} new); valid {len(valid)}")
    for k, v in c.most_common():
        print(f"  {v:6d} {k}")


if __name__ == "__main__":
    main()
