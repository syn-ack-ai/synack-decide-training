"""v8 ops / log slice: decision records from public log datasets (docs/V6_PLAN.md, "v8 (future planning)").

Gold-labelled tasks
  ops/template      Loghub-2.0 (CC-BY-4.0): which event template produced this raw log line? (choice, 6-8 options;
                    distractors are the most similar templates of the same system)
  ops/same_event    Loghub-2.0: were these two lines produced by the same logging statement? (yes/no, hard negatives)
  ops/bgl_window    Loghub BGL (CC-BY-4.0): does this window of 10 lines contain an alert? (yes/no) and which line is
                    the first alert? (choice over the 10 lines, positive windows only)
  ops/hdfs_session  Loghub HDFS_v1: is this block's session anomalous? (yes/no, sessions of <= 40 lines)
  ops/rca           RCAEval RE2/RE3 (CC-BY-4.0): which service is the root cause (and, for RE2, which fault type), from
                    a summary of logs and metrics 10 min before / 5 min after the fault
  ops/hadoop_fault  Loghub Hadoop: which fault was injected into this MapReduce job? (WARN/ERROR/FATAL lines, 3 views)
Teacher-only task (no gold; labelled later by Kimi-K3 + Gemma-4-31B through v2/teacher_api.py --no-gold)
  ops/triage        one raw line per event template of every Loghub-2.0 system: severity, subsystem, act?

Records use the v2 decision-record shape (v2/common.py). Run where the data is:
  python build_ops.py --data ~/opsdata --out ~/opsdata/out
"""
import argparse
import collections
import csv
import json
import random
import re
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

SYSTEMS = {  # Loghub-2.0 system -> what the machine is (state context for every row)
    "Apache": "Apache HTTP server error log", "OpenSSH": "OpenSSH server log on a Linux host",
    "Proxifier": "Proxifier proxy client log on a Windows desktop", "HDFS": "Hadoop Distributed File System (HDFS) cluster log",
    "Linux": "Linux server system log (syslog)", "HealthApp": "Android health-tracking app log",
    "HPC": "high-performance computing cluster log", "Zookeeper": "Apache ZooKeeper coordination service log",
    "Thunderbird": "Thunderbird supercomputer system log (Sandia National Labs)", "OpenStack": "OpenStack cloud compute (Nova) log",
    "BGL": "Blue Gene/L supercomputer RAS log (Lawrence Livermore)", "Mac": "macOS laptop system log",
    "Hadoop": "Hadoop MapReduce job log", "Spark": "Apache Spark job log",
}
TOK = re.compile(r"[A-Za-z]+")
LABELLED = ("BGL", "Thunderbird")

TRIAGE_Q = {
    "severity": {"type": "choice", "instructions": "How serious is this log entry for the health of the system that wrote it?",
                 "criteria": {"noise": "Harmless or expected: routine status, boot or startup chatter, debug output, benign quirks.",
                              "minor": "A real but low-impact problem that can wait.",
                              "important": "Likely breaks a feature or degrades the system; should be fixed soon.",
                              "critical": "Data loss, hardware failure, security breach or an outage."}},
    "subsystem": {"type": "choice", "instructions": "Which part of the system is this log entry about?",
                  "criteria": {"hardware": "CPU, memory, GPU, power, fans, temperature or other hardware",
                               "storage": "Disks, filesystems, block storage, mounts",
                               "network": "Network interfaces, connections, DNS, proxies, RPC",
                               "security": "Authentication, authorization, firewall, certificates",
                               "kernel": "Kernel, firmware, drivers, boot",
                               "service": "A system service, daemon, scheduler or cluster manager",
                               "application": "A user-level application, job or task",
                               "other": "Something else"}},
    "act": {"type": "noul", "instructions": "Should an operator do something about this log entry?"},
}


def toks(s):
    return set(w.lower() for w in TOK.findall(s))


def jaccard(a, b):
    return len(a & b) / (len(a | b) or 1)


def load_system(root, name):
    d = root / "loghub2" / "x" / name
    raw = (d / f"{name}_full.log").read_text(errors="replace").splitlines()
    if name in LABELLED:  # first field is the anomaly label ("-" or an alert code): drop it so it cannot leak
        raw = [l.split(" ", 1)[1] if " " in l else l for l in raw]
    rows = list(csv.DictReader(open(d / f"{name}_full.log_structured.csv", errors="replace")))
    tmpl = {r["EventId"]: r["EventTemplate"] for r in csv.DictReader(open(d / f"{name}_full.log_templates.csv", errors="replace"))}
    by_event = collections.defaultdict(list)
    for r in rows:
        i = int(r["LineId"]) - 1
        if 0 <= i < len(raw) and r["EventId"] in tmpl:
            by_event[r["EventId"]].append(raw[i][:1500])
    return tmpl, by_event


def tmpl_re(t):
    return re.compile(".*?".join(re.escape(part) for part in t.split("<*>")), re.S)


def fits(rx, line):
    try:
        return rx.search(line) is not None
    except RecursionError:
        return False


def template_records(name, tmpl, by_event, rng, per_event=3, n_opts=(6, 8)):
    out = []
    tt = {e: toks(t) for e, t in tmpl.items()}
    rx = {e: tmpl_re(t) for e, t in tmpl.items()}
    for e, lines in by_event.items():
        if len(tmpl) < 4:
            break
        for k, line in enumerate(rng.sample(lines, min(per_event, len(lines)))):
            # a distractor whose pattern also fits the line would make the question ambiguous (near-duplicate templates)
            sim = sorted((x for x in tmpl if x != e and not fits(rx[x], line)), key=lambda x: -jaccard(tt[e], tt[x]))
            n = min(rng.randint(*n_opts), len(tmpl))
            hard = sim[:max(1, (n - 1) * 2 // 3)]
            rest = [x for x in sim if x not in hard]
            opts = hard + rng.sample(rest, max(0, min(len(rest), n - 1 - len(hard)))) + [e]
            if len(opts) < 3:
                continue
            rng.shuffle(opts)
            out.append({"id": f"ops/template:{name}:{e}:{k}", "source": "ops/template",
                        "state": {"system": SYSTEMS[name], "log_line": line},
                        "questions": {"q": {"type": "choice",
                                            "instructions": "Which event template produced this raw log line? In a template, <*> stands "
                                                            "for a variable part such as a number, ID, path or host name.",
                                            "criteria": {o: tmpl[o] for o in opts}}},
                        "expected": {"q": e}})
    return out


def same_event_records(name, tmpl, by_event, rng, n):
    tt = {e: toks(t) for e, t in tmpl.items()}
    multi = [e for e, l in by_event.items() if len(set(l)) >= 2]
    out = []
    for k in range(n):
        if not multi:
            break
        e = rng.choice(multi)
        a = rng.choice(by_event[e])
        if k % 2 == 0:
            b = rng.choice([x for x in set(by_event[e]) if x != a] or [a])
            same = True
        else:
            near = sorted((x for x in by_event if x != e), key=lambda x: -jaccard(tt[e], tt[x]))[:3]
            if not near:
                continue
            e2 = rng.choice(near)
            b = rng.choice(by_event[e2])
            if fits(tmpl_re(tmpl[e]), b) or fits(tmpl_re(tmpl[e2]), a):
                continue
            same = False
        out.append({"id": f"ops/same_event:{name}:{k}", "source": "ops/same_event",
                    "state": {"system": SYSTEMS[name], "line_a": a, "line_b": b},
                    "questions": {"q": {"type": "noul",
                                        "instructions": "Were these two log lines produced by the same logging statement, differing "
                                                        "only in variable values such as numbers, IDs, paths or host names?"}},
                    "expected": {"q": same}})
    return out


def triage_records(name, tmpl, by_event, rng):
    out = []
    for e, lines in by_event.items():
        line = rng.choice(lines)
        out.append({"id": f"ops/triage:{name}:{e}", "source": "ops/triage",
                    "state": {"system": SYSTEMS[name], "log_entry": line},
                    "questions": TRIAGE_Q, "expected": {}})
    return out


def bgl_records(root, rng, n_windows, win=10):
    path = root / "loghub1" / "x" / "BGL.log"
    lines = path.read_text(errors="replace").splitlines()
    label = [l.split(" ", 1)[0] != "-" for l in lines]
    body = [l.split(" ", 1)[1] if " " in l else l for l in lines]
    pos = [i for i in range(0, len(lines) - win, win) if any(label[i:i + win])]
    neg = [i for i in range(0, len(lines) - win, win) if not any(label[i:i + win])]
    out = []
    for k, i in enumerate(rng.sample(pos, min(n_windows // 2, len(pos))) + rng.sample(neg, min(n_windows // 2, len(neg)))):
        w = body[i:i + win]
        has = any(label[i:i + win])
        qs = {"alert": {"type": "noul", "instructions": "Does this window of consecutive log lines contain at least one alert, "
                                                        "i.e. a line reporting a real fault rather than routine activity?"}}
        exp = {"alert": has}
        if has:
            first = next(j for j in range(win) if label[i + j])
            qs["which"] = {"type": "choice", "instructions": "Which line is the first alert in this window?",
                           "criteria": {f"line_{j + 1}": w[j][:600] for j in range(win)}}
            exp["which"] = f"line_{first + 1}"
        out.append({"id": f"ops/bgl_window:{i}", "source": "ops/bgl_window",
                    "state": {"system": SYSTEMS["BGL"], "log_window": [f"{j + 1}: {w[j][:600]}" for j in range(win)]},
                    "questions": qs, "expected": exp})
    return out


def hdfs_records(root, rng, n):
    d = root / "loghub1" / "x"
    lab = {r["BlockId"]: r["Label"] == "Anomaly" for r in csv.DictReader(open(next(d.rglob("anomaly_label.csv"))))}
    sess = collections.defaultdict(list)
    blk = re.compile(r"blk_-?\d+")
    with open(next(d.rglob("HDFS.log")), errors="replace") as f:
        for line in f:
            for b in set(blk.findall(line)):
                if b in lab and len(sess[b]) <= 41:
                    sess[b].append(line.rstrip()[:400])
    ok = [b for b, l in sess.items() if len(l) <= 40]
    pos = [b for b in ok if lab[b]]
    neg = [b for b in ok if not lab[b]]
    out = []
    for b in rng.sample(pos, min(n // 2, len(pos))) + rng.sample(neg, min(n // 2, len(neg))):
        out.append({"id": f"ops/hdfs_session:{b}", "source": "ops/hdfs_session",
                    "state": {"system": SYSTEMS["HDFS"], "block": b, "session_log": sess[b]},
                    "questions": {"q": {"type": "noul", "instructions": "Is this block's session anomalous: did the block's "
                                                                        "write, replication or deletion go wrong?"}},
                    "expected": {"q": lab[b]}})
    return out


HADOOP_FAULTS = {"normal": "No fault: the job ran normally", "machine_down": "A worker machine went down",
                 "network_disconnection": "A worker lost its network connection", "disk_full": "A worker's disk filled up"}


def hadoop_records(root, rng, views=3, max_lines=40):
    d = root / "loghub1" / "x"
    lab, cur = {}, None
    for line in open(d / "abnormal_label.txt"):
        line = line.strip()
        key = {"Normal:": "normal", "Machine down:": "machine_down", "Network disconnection:": "network_disconnection",
               "Disk full:": "disk_full"}.get(line)
        if key:
            cur = key
        elif line.startswith("+ application_") and cur:
            lab[line[2:]] = cur
    level = re.compile(r" (WARN|ERROR|FATAL) ")
    out = []
    for app, fault in sorted(lab.items()):
        lines = []
        for f in sorted((d / app).glob("*.log")):
            lines += [f"{f.stem.split('_')[-1]}: {l.rstrip()[:300]}" for l in open(f, errors="replace") if level.search(l)]
        if not lines:
            lines = ["(no WARN, ERROR or FATAL lines in any container log)"]
        for v in range(views):
            pick = sorted(rng.sample(range(len(lines)), min(max_lines, len(lines))))
            out.append({"id": f"ops/hadoop_fault:{app}:{v}", "source": "ops/hadoop_fault",
                        "state": {"system": SYSTEMS["Hadoop"], "application": app,
                                  "warnings_and_errors": [lines[i] for i in pick],
                                  "total_warning_error_lines": len(lines)},
                        "questions": {"q": {"type": "choice", "instructions": "Which fault, if any, was injected while this "
                                                                              "MapReduce job ran?", "criteria": HADOOP_FAULTS}},
                        "expected": {"q": fault}})
    return out


RE2_FAULTS = {"cpu": "CPU hog", "mem": "Memory leak or memory hog", "disk": "Disk I/O stress", "delay": "Network delay",
              "loss": "Network packet loss", "socket": "Socket exhaustion"}
ERRORISH = re.compile(r"error|exception|fail|timeout|timed out|refused|panic|unavailable|denied|reset|50[0-9]", re.I)
NUM = re.compile(r"[0-9a-f]{6,}|\d+")


def rca_case(case_dir, before=600, after=300):
    inject = int(float((case_dir / "inject_time.txt").read_text().split()[0]))
    logs = collections.defaultdict(lambda: {"before": 0, "after": 0, "err_before": 0, "err_after": 0,
                                            "seen": set(), "new": collections.Counter(), "example": {}})
    with open(case_dir / "logs.csv", errors="replace") as f:
        for r in csv.DictReader(f):
            try:
                t = int(r["timestamp"]) / 1e9
            except (KeyError, ValueError):
                continue
            msg = (r.get("message") or "").strip()
            s = logs[r["container_name"]]
            key = NUM.sub("#", msg)[:160]
            if inject - before <= t < inject:
                s["before"] += 1
                s["err_before"] += bool(ERRORISH.search(msg))
                s["seen"].add(key)
            elif inject <= t < inject + after:
                s["after"] += 1
                s["err_after"] += bool(ERRORISH.search(msg))
                if key not in s["seen"]:
                    s["new"][key] += 1
                    s["example"].setdefault(key, msg[:240])
    log_summary, new_msgs = {}, []
    for c, s in logs.items():
        if s["before"] + s["after"] == 0:
            continue
        log_summary[c] = {"lines_per_min_before": round(s["before"] / (before / 60), 1),
                          "lines_per_min_after": round(s["after"] / (after / 60), 1),
                          "error_lines_per_min_before": round(s["err_before"] / (before / 60), 1),
                          "error_lines_per_min_after": round(s["err_after"] / (after / 60), 1)}
        for k, n in s["new"].most_common(3):
            new_msgs.append((bool(ERRORISH.search(k)), n, c, s["example"][k]))
    new_msgs.sort(key=lambda x: (-x[0], -x[1]))
    rows = list(csv.reader(open(case_dir / "simple_metrics.csv")))
    head, data = rows[0], rows[1:]
    moves = []
    for j, col in enumerate(head[1:], 1):
        b_, a_ = [], []
        for r in data:
            try:
                t, v = float(r[0]), float(r[j])
            except (ValueError, IndexError):
                continue
            (b_ if inject - before <= t < inject else a_ if inject <= t < inject + after else []).append(v)
        if len(b_) < 10 or len(a_) < 10:
            continue
        mb, ma = sum(b_) / len(b_), sum(a_) / len(a_)
        sd = (sum((x - mb) ** 2 for x in b_) / len(b_)) ** 0.5
        z = (ma - mb) / (sd + 1e-9 + 0.01 * abs(mb))
        moves.append((abs(z), col, mb, ma))
    moves.sort(reverse=True)
    services = sorted(set(log_summary) | {c.rsplit("_", 1)[0] for c in head[1:]})
    state = {"incident": "An alert fired; the fault started at minute 0. Windows: the 10 minutes before and 5 minutes after.",
             "biggest_metric_changes": [f"{c}: {mb:.3g} -> {ma:.3g}" for _, c, mb, ma in moves[:12]],
             "logs_per_service": log_summary,
             "new_log_messages_after_fault": [f"{c} (x{n}): {m}" for _, n, c, m in new_msgs[:15]]}
    return services, state


def rcaeval_records(root):
    out = []
    for case in sorted((root / "rcaeval" / "x").glob("RE*/RE*-*/*/*")):
        if not (case / "inject_time.txt").exists():
            continue
        name = case.parent.name
        svc, fault = name.rsplit("_", 1)
        system = {"OB": "Online Boutique (Google microservice demo shop)", "SS": "Sock Shop (microservice demo shop)",
                  "TT": "Train Ticket (microservice train booking system)"}[case.parent.parent.name.split("-")[1]]
        services, state = rca_case(case)
        if svc not in services:
            continue
        qs = {"root_cause": {"type": "choice", "instructions": "Which service is the root cause of this incident?",
                             "criteria": {s: None for s in services}}}
        exp = {"root_cause": svc}
        if fault in RE2_FAULTS:
            qs["fault"] = {"type": "choice", "instructions": "What kind of fault caused this incident?", "criteria": RE2_FAULTS}
            exp["fault"] = fault
        out.append({"id": f"ops/rca:{case.parent.parent.name}:{name}:{case.name}", "source": "ops/rca",
                    "state": {"system": system, **state}, "questions": qs, "expected": exp})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20261006)
    p.add_argument("--systems", default=",".join(SYSTEMS))
    p.add_argument("--only-loghub2", action="store_true")
    a = p.parse_args()
    rng = random.Random(a.seed)
    root, out = Path(a.data).expanduser(), Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    recs = collections.defaultdict(list)
    for name in a.systems.split(","):
        tmpl, by_event = load_system(root, name)
        recs["template"] += template_records(name, tmpl, by_event, rng)
        recs["same_event"] += same_event_records(name, tmpl, by_event, rng, n=150)
        recs["triage"] += triage_records(name, tmpl, by_event, rng)
        print(f"{name}: {len(tmpl)} templates, {sum(map(len, by_event.values()))} lines", flush=True)
    if not a.only_loghub2:
        recs["bgl_window"] = bgl_records(root, rng, 3000)
        recs["hdfs_session"] = hdfs_records(root, rng, 2000)
        recs["hadoop_fault"] = hadoop_records(root, rng)
        recs["rca"] = rcaeval_records(root)
    for k, v in recs.items():
        with open(out / f"{k}.jsonl", "w") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{k}: {len(v)} records", flush=True)


if __name__ == "__main__":
    main()
