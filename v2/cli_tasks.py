"""Verifiable CLI tasks for stage 3 of the E4B experiment (run in the stage-3 sandbox, v2/sandbox_runner.py).

Every task family builds a randomised fixture (files, a git repo, a JSON or SQLite file) and checks the RESULT: the
command's stdout (saved by the runner in /tmp/out), file contents or repository state, never the command's text. Each
family carries a known-correct and a known-wrong command, so `python cli_tasks.py --selftest` proves the check accepts
the right answer and rejects the wrong one before anything is trained or scored on it.

Families marked heldout=True are never used for training (they are the CLI test).

  python cli_tasks.py --selftest            # meta-test every family in the sandbox
  python cli_tasks.py --make N --split train|test --out tasks.jsonl
"""
import argparse
import json
import random
import string
import urllib.request

RUNNER = "http://127.0.0.1:18899/run"
OUT_IS = 'test "$(cat /tmp/out)" = "$(cat /tmp/expected)"'  # stdout must equal the expected answer exactly
OUT_INT = 'test "$(tr -d "[:space:]" < /tmp/out)" = "$(cat /tmp/expected)"'  # a number, whitespace ignored


def word(rng, n=6):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


def git_repo(cmds):
    return ("git init -q repo && cd repo && git config user.email dev@example.com && git config user.name dev && "
            + " && ".join(cmds))


FAMILIES = {}


def family(name, heldout=False):
    def deco(fn):
        FAMILIES[name] = (fn, heldout)
        return fn
    return deco


@family("count_ext")
def count_ext(rng):
    ext = rng.choice(["py", "md", "json", "txt", "log"])
    others = [e for e in ["py", "md", "json", "txt", "log", "yaml"] if e != ext]
    n = rng.randint(3, 12)
    files = [f"{rng.choice(['src', 'lib', 'docs', 'a/b', 'a/b/c', '.'])}/{word(rng)}.{ext}" for _ in range(n)]
    noise = [f"{rng.choice(['src', 'x'])}/{word(rng)}.{rng.choice(others)}" for _ in range(rng.randint(3, 8))]
    setup = " && ".join(f"mkdir -p $(dirname {f}) && echo x > {f}" for f in files + noise) + f" && echo {len(set(files))} > /tmp/expected"
    return {"question": f"How many .{ext} files are there in the current directory, including all subdirectories? Print just the number.",
            "setup": setup, "check": OUT_INT,
            "right": f'find . -type f -name "*.{ext}" | wc -l', "wrong": f'ls *.{ext} | wc -l'}


@family("largest_file")
def largest_file(rng):
    sizes = rng.sample(range(1000, 90000, 137), 6)
    names = [f"{rng.choice(['data', 'logs', 'tmp'])}/{word(rng)}.bin" for _ in sizes]
    setup = " && ".join(f"mkdir -p $(dirname {n}) && head -c {s} /dev/zero > {n}" for n, s in zip(names, sizes))
    big = names[sizes.index(max(sizes))]
    setup += f" && mkdir -p bigdir && head -c {max(sizes) + 5000} /dev/zero > bigdir/.hidden_cache && echo ./{big} > /tmp/expected"
    setup = setup.replace("bigdir/.hidden_cache", "bigdir/notes")  # a bigger file in another folder: only data/logs/tmp count
    big_all = "./bigdir/notes"
    setup = setup.replace(f"echo ./{big} > /tmp/expected", f"echo {big_all} > /tmp/expected")
    return {"question": "Print the path (as find prints it, starting with ./) of the single largest regular file under the current directory, recursively.",
            "setup": setup, "check": OUT_IS,
            "right": "find . -type f -printf '%s %p\\n' | sort -n | tail -1 | cut -d' ' -f2-",
            "wrong": "ls -S | head -1"}


@family("grep_count")
def grep_count(rng):
    w = word(rng, 5)
    lines = []
    k = 0
    for _ in range(rng.randint(20, 60)):
        if rng.random() < 0.3:
            lines.append(f"{word(rng)} {w.upper() if rng.random() < 0.5 else w} {word(rng)}")
            k += 1
        else:
            lines.append(f"{word(rng)} {word(rng)}")
    body = "\\n".join(lines)
    return {"question": f"How many lines of app.log contain the word '{w}', ignoring case? Print just the number.",
            "setup": f"printf '{body}\\n' > app.log && echo {k} > /tmp/expected", "check": OUT_INT,
            "right": f"grep -ic '{w}' app.log", "wrong": f"grep -c '{w}' app.log"}


@family("sum_column")
def sum_column(rng):
    rows = [(word(rng), rng.randint(1, 500)) for _ in range(rng.randint(5, 20))]
    body = "name,amount\\n" + "\\n".join(f"{a},{b}" for a, b in rows)
    return {"question": "sales.csv has a header row and columns name,amount. Print the sum of the amount column as an integer.",
            "setup": f"printf '{body}\\n' > sales.csv && echo {sum(b for _, b in rows)} > /tmp/expected", "check": OUT_INT,
            "right": "awk -F, 'NR>1{s+=$2} END{print s}' sales.csv", "wrong": "awk -F, '{s+=$2} END{print s+1}' sales.csv"}


@family("top_words")
def top_words(rng):
    vocab = [word(rng, 4) for _ in range(8)]
    counts = sorted(rng.sample(range(2, 30), 8), reverse=True)
    words = [v for v, c in zip(vocab, counts) for _ in range(c)]
    rng.shuffle(words)
    body = "\\n".join(words)
    top3 = "\\n".join(vocab[:3])
    return {"question": "words.txt has one word per line. Print the 3 most frequent words, most frequent first, one per line, words only.",
            "setup": f"printf '{body}\\n' > words.txt && printf '{top3}\\n' > /tmp/expected", "check": OUT_IS,
            "right": "sort words.txt | uniq -c | sort -rn | head -3 | awk '{print $2}'", "wrong": "sort words.txt | uniq | head -3"}


@family("replace_inplace")
def replace_inplace(rng):
    old, new = word(rng, 5), word(rng, 5)
    files = [f"conf/{word(rng)}.ini" for _ in range(rng.randint(2, 4))]
    setup = "mkdir -p conf && " + " && ".join(f"printf 'host={old}\\nbackup={old}.bak\\n' > {f}" for f in files)
    check = f"! grep -rq '{old}' conf && test $(grep -rho '{new}' conf | wc -l) -eq {2 * len(files)}"
    return {"question": f"Replace every occurrence of '{old}' with '{new}' in all .ini files under conf/, editing the files in place.",
            "setup": setup, "check": check,
            "right": f"sed -i 's/{old}/{new}/g' conf/*.ini", "wrong": f"sed 's/{old}/{new}/g' conf/*.ini"}


@family("tar_exclude")
def tar_exclude(rng):
    keep = [f"project/{word(rng)}.js" for _ in range(3)]
    setup = ("mkdir -p project/node_modules/x project/sub/node_modules && " + " && ".join(f"echo x > {k}" for k in keep)
             + " && echo dep > project/node_modules/x/i.js && echo dep > project/sub/node_modules/j.js")
    check = "tar -tzf project.tar.gz > /tmp/l && ! grep -q node_modules /tmp/l && " + " && ".join(f"grep -q '{k}' /tmp/l" for k in keep)
    return {"question": "Create project.tar.gz from the project/ directory, excluding every node_modules directory at any depth.",
            "setup": setup, "check": check,
            "right": "tar -czf project.tar.gz --exclude='node_modules' project/", "wrong": "tar -czf project.tar.gz project/"}


@family("json_field")
def json_field(rng):
    users = [{"name": word(rng), "age": rng.randint(18, 80), "active": rng.random() < 0.5} for _ in range(rng.randint(4, 9))]
    act = "\\n".join(u["name"] for u in users if u["active"])
    if not act:
        users[0]["active"] = True
        act = users[0]["name"]
    return {"question": "users.json is a JSON array of objects with name, age and active fields. Print the names of the active users, one per line, in file order.",
            "setup": f"echo '{json.dumps(users)}' > users.json && printf '{act}\\n' > /tmp/expected", "check": OUT_IS,
            "right": "jq -r '.[] | select(.active) | .name' users.json", "wrong": "jq -r '.[].name' users.json"}


@family("git_soft_reset")
def git_soft_reset(rng):
    f = word(rng) + ".txt"
    setup = git_repo([f"echo one > {f}", f"git add {f}", "git commit -qm first", f"echo two >> {f}", f"git add {f}", "git commit -qm second"])
    return {"question": "In the git repository repo/, undo the last commit but keep its changes staged.",
            "setup": setup, "check": f'cd repo && test "$(git log --oneline | wc -l)" -eq 1 && git diff --cached --name-only | grep -qx {f}',
            "right": "cd repo && git reset --soft HEAD~1", "wrong": "cd repo && git reset HEAD~1"}


@family("git_unstage")
def git_unstage(rng):
    a, b = word(rng) + ".py", word(rng) + ".py"
    setup = git_repo([f"echo x > {a}", f"echo y > {b}", f"git add {a} {b}", "git commit -qm init", f"echo z >> {a}", f"echo z >> {b}", f"git add {a} {b}"])
    check = (f"cd repo && git diff --cached --name-only | grep -qx {b} && ! git diff --cached --name-only | grep -qx {a} "
             f"&& grep -q z {a}")
    return {"question": f"In repo/, both {a} and {b} have staged changes. Unstage {a} only, keeping its edits in the working tree.",
            "setup": setup, "check": check,
            "right": f"cd repo && git restore --staged {a}", "wrong": f"cd repo && git checkout -- {a}"}


@family("git_changed_files", heldout=True)
def git_changed_files(rng):
    files = [word(rng) + ".md" for _ in range(4)]
    changed = sorted(rng.sample(files, 2))
    setup = git_repo([*(f"echo a > {f}" for f in files), "git add .", "git commit -qm one",
                      *(f"echo b >> {f}" for f in changed), "git add .", "git commit -qm two"])
    setup += " && cd /work && printf '" + "\\n".join(changed) + "\\n' > /tmp/expected"
    return {"question": "In repo/, print the names of the files changed in the most recent commit, one per line, sorted.",
            "setup": setup, "check": OUT_IS,
            "right": "cd repo && git diff --name-only HEAD~1 HEAD | sort", "wrong": "cd repo && git ls-files | sort"}


@family("git_amend_msg", heldout=True)
def git_amend_msg(rng):
    msg = "Fix " + word(rng) + " handling"
    setup = git_repo(["echo a > f", "git add f", "git commit -qm wip"])
    return {"question": f"In repo/, change the message of the last commit to \"{msg}\" without changing its content or adding a commit.",
            "setup": setup, "check": f'cd repo && test "$(git log -1 --format=%s)" = "{msg}" && test "$(git log --oneline | wc -l)" -eq 1',
            "right": f"cd repo && git commit --amend -m '{msg}'", "wrong": f"cd repo && git commit --allow-empty -m '{msg}'"}


@family("sqlite_count", heldout=True)
def sqlite_count(rng):
    rows = [(word(rng), rng.choice(["open", "closed", "pending"])) for _ in range(rng.randint(8, 25))]
    ins = ";".join(f"insert into t values('{a}','{b}')" for a, b in rows)
    k = sum(b == "open" for _, b in rows)
    return {"question": "app.db is a SQLite database with a table t(name, status). Print how many rows have status 'open'.",
            "setup": f"sqlite3 app.db \"create table t(name text, status text);{ins}\" && echo {k} > /tmp/expected", "check": OUT_INT,
            "right": "sqlite3 app.db \"select count(*) from t where status='open'\"", "wrong": "sqlite3 app.db 'select count(*) from t'"}


@family("dedupe_keep_order", heldout=True)
def dedupe_keep_order(rng):
    items = [word(rng, 4) for _ in range(6)]
    seq = [rng.choice(items) for _ in range(20)]
    out, seen = [], set()
    for s in seq:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return {"question": "list.txt has one item per line with duplicates. Print each item once, keeping the order of first appearance.",
            "setup": "printf '" + "\\n".join(seq) + "\\n' > list.txt && printf '" + "\\n".join(out) + "\\n' > /tmp/expected", "check": OUT_IS,
            "right": "awk '!seen[$0]++' list.txt", "wrong": "sort -u list.txt"}


@family("make_executable", heldout=True)
def make_executable(rng):
    sh = [f"scripts/{word(rng)}.sh" for _ in range(3)]
    other = f"scripts/{word(rng)}.txt"
    setup = "mkdir -p scripts && " + " && ".join(f"echo 'echo hi' > {f}" for f in sh + [other])
    check = " && ".join(f"test -x {f}" for f in sh) + f" && ! test -x {other}"
    return {"question": "Make every .sh file in scripts/ executable, and nothing else.",
            "setup": setup, "check": check, "right": "chmod +x scripts/*.sh", "wrong": "chmod +x scripts/*"}


def run(**kw):
    req = urllib.request.Request(RUNNER, data=json.dumps(kw).encode())
    return json.load(urllib.request.urlopen(req, timeout=180))


def selftest(seed=0):
    rng = random.Random(seed)
    bad = 0
    for name, (fn, held) in FAMILIES.items():
        t = fn(rng)
        r_ok = run(setup=t["setup"], command=t["right"], check=t["check"])
        r_bad = run(setup=t["setup"], command=t["wrong"], check=t["check"])
        ok = r_ok["check_passed"] and not r_bad["check_passed"]
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {name:20s} {'(held out)' if held else '          '} right passes: {r_ok['check_passed']}, "
              f"wrong fails: {not r_bad['check_passed']}" + ("" if ok else f" | right stderr: {r_ok['stderr'][:120]!r}"), flush=True)
    print(f"{len(FAMILIES) - bad}/{len(FAMILIES)} families verified")


def make(n, split, seed):
    rng = random.Random(seed)
    names = [k for k, (_, h) in FAMILIES.items() if h == (split == "test")]
    out = []
    for i in range(n):
        name = names[i % len(names)]
        t = FAMILIES[name][0](rng)
        out.append({"id": f"cli:{split}:{name}:{i}", "family": name, **{k: t[k] for k in ("question", "setup", "check", "right")}})
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--make", type=int)
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out")
    a = p.parse_args()
    if a.selftest:
        selftest()
    if a.make:
        rows = make(a.make, a.split, a.seed)
        with open(a.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{len(rows)} {a.split} tasks -> {a.out}")
