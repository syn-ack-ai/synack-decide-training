"""Code-verified synthetic decision records (fresh items; answers computed, never copied).

Families mirror reasoning skills the index probes but which have no legal train split:
boolean logic, date arithmetic, navigation, shuffled-object tracking, web of lies, logical
ordering, temporal sequences, multistep arithmetic, and CRUXEval-style code output prediction.
Usage: python build_synthetic.py --out synthetic.jsonl --per-family 1500
"""
import argparse
import datetime as dt
import json
import random

PEOPLE = ["Alice", "Bob", "Claire", "Dave", "Eve", "Fred", "Gertrude", "Hank", "Ivy", "Jamal", "Kofi", "Lena"]
ITEMS = ["a red ball", "a blue present", "a pink ball", "a green book", "an orange scarf", "a black hat",
         "a white kite", "a purple frisbee", "a yellow umbrella", "a brown bag"]
RECORDS = []


def add(fam, i, state, instr, criteria, gold, rng):
    keys = list(criteria)
    rng.shuffle(keys)
    RECORDS.append({"id": f"synthetic:{fam}:{i}", "source": f"synthetic/{fam}", "state": state,
                    "questions": {"q": {"type": "choice", "instructions": instr,
                                        "criteria": {k: criteria[k] for k in keys}}},
                    "expected": {"q": gold}})


def boolean(rng, i):
    def expr(d):
        if d == 0 or rng.random() < 0.3:
            return rng.choice(["True", "False"])
        op = rng.choice(["and", "or", "not"])
        return f"not {expr(d - 1)}" if op == "not" else f"( {expr(d - 1)} {op} {expr(d - 1)} )"
    e = expr(rng.randint(2, 4))
    v = eval(e)
    add("boolean", i, e + " is", "Which option is the correct answer?", {"True": "True", "False": "False"}, str(v), rng)


def dates(rng, i):
    base = dt.date(rng.randint(1950, 2045), rng.randint(1, 12), rng.randint(1, 28))
    n, unit = rng.randint(1, 60), rng.choice(["days", "weeks"])
    sign = rng.choice([1, -1])
    delta = dt.timedelta(days=n * (7 if unit == "weeks" else 1)) * sign
    ans = base + delta
    word = "later" if sign > 0 else "earlier"
    q = f"Today is {base:%m/%d/%Y}. What is the date {n} {unit} {word} in MM/DD/YYYY?"
    opts = {ans}
    while len(opts) < 6:
        opts.add(ans + dt.timedelta(days=rng.choice([-365, -30, -7, -1, 1, 7, 30, 365, -2, 2])))
    crit = {f"{d:%m/%d/%Y}": f"{d:%m/%d/%Y}" for d in opts}
    add("dates", i, q, "Which option is the correct answer?", crit, f"{ans:%m/%d/%Y}", rng)


def navigate(rng, i):
    x = y = 0
    facing = 0
    steps = []
    for _ in range(rng.randint(3, 8)):
        if rng.random() < 0.3:
            t = rng.choice(["Turn left.", "Turn right.", "Turn around."])
            facing = (facing + {"Turn left.": 3, "Turn right.": 1, "Turn around.": 2}[t]) % 4
            steps.append(t)
        else:
            k = rng.randint(1, 9)
            dx, dy = [(0, 1), (1, 0), (0, -1), (-1, 0)][facing]
            x, y = x + dx * k, y + dy * k
            steps.append(f"Take {k} step{'s' if k > 1 else ''}.")
    q = "If you follow these instructions, do you return to the starting point? Always face forward. " + " ".join(steps)
    add("navigate", i, q, "Which option is the correct answer?", {"Yes": "Yes", "No": "No"},
        "Yes" if (x, y) == (0, 0) else "No", rng)


def tracking(rng, i):
    n = rng.choice([3, 5, 7])
    ppl = rng.sample(PEOPLE, n)
    items = rng.sample(ITEMS, n)
    hold = dict(zip(ppl, items))
    swaps = []
    for _ in range(n):
        a, b = rng.sample(ppl, 2)
        hold[a], hold[b] = hold[b], hold[a]
        swaps.append(f"{a} and {b} swap their gifts.")
    who = rng.choice(ppl)
    intro = ", ".join(f"{p} has {it}" for p, it in zip(ppl, items))
    q = f"{intro}. As the event goes on, pairs trade gifts. " + " ".join(swaps) + f" At the end, {who} has"
    add("tracking", i, q, "Which option is the correct answer?", {it: it for it in items}, hold[who], rng)


def web_of_lies(rng, i):
    ppl = rng.sample(PEOPLE, rng.randint(3, 6))
    truth = rng.choice([True, False])
    parts = [f"{ppl[0]} {'tells the truth' if truth else 'lies'}."]
    for a, b in zip(ppl, ppl[1:]):
        says = rng.choice([True, False])
        parts.append(f"{b} says {a} {'tells the truth' if says else 'lies'}.")
        truth = truth == says
    q = " ".join(parts) + f" Does {ppl[-1]} tell the truth?"
    add("web_of_lies", i, q, "Which option is the correct answer?", {"Yes": "Yes", "No": "No"},
        "Yes" if truth else "No", rng)


def ordering(rng, i):
    n = rng.choice([3, 5, 7])
    objs = rng.sample(["the teal kayak", "the cedar crate", "the brass lamp", "the wool rug", "the clay vase",
                       "the steel drum", "the linen chest", "the jade bowl", "the cork board", "the slate tile"], n)
    order = objs[:]
    rng.shuffle(order)
    pos = {o: k for k, o in enumerate(order)}
    facts = set()
    while True:
        a, b = rng.sample(objs, 2)
        facts.add(f"{a.capitalize()} is to the {'left' if pos[a] < pos[b] else 'right'} of {b}.")
        k = rng.randrange(n)
        if rng.random() < 0.3:
            facts.add(f"{order[k].capitalize()} is the {['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh'][k]} from the left.")
        import itertools
        ok = [p for p in itertools.permutations(objs) if all(check(f, p) for f in facts)]
        if len(ok) == 1:
            break
    k = rng.randrange(n)
    ordinal = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"][k]
    q = f"In a storage row there are {n} items: {', '.join(objs)}. " + " ".join(sorted(facts)) + \
        f" Which item is the {ordinal} from the left?"
    add("ordering", i, q, "Which option is the correct answer?", {o: o.capitalize() for o in objs}, order[k], rng)


def check(fact, perm):
    pos = {o: k for k, o in enumerate(perm)}
    low = fact[0].lower() + fact[1:]
    if " is to the " in low:
        a, rest = low.split(" is to the ", 1)
        side, b = rest.split(" of ", 1)
        b = b.rstrip(".")
        return (pos[a] < pos[b]) if side == "left" else (pos[a] > pos[b])
    a, rest = low.split(" is the ", 1)
    k = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh"].index(rest.split(" ")[0])
    return pos[a] == k


def temporal(rng, i):
    who = rng.choice(PEOPLE)
    hours = list(range(5, 22))
    start = rng.randint(0, len(hours) - 8)
    span = hours[start:start + rng.randint(5, 8)]
    places = rng.sample(["the bakery", "the gym", "the library", "the park", "the museum", "the beach",
                         "the office", "the market", "the cafe"], 4)
    free = rng.randrange(len(span) - 1)
    seen, acts = [], []
    cur = 0
    for k in range(len(span) - 1):
        if k == free:
            continue
        p = rng.choice(places[1:])
        seen.append(f"From {span[k]}:00 to {span[k + 1]}:00, someone saw {who} at {p}.")
    q = (f"{who} woke up at {span[0]}:00. " + " ".join(seen) +
         f" {places[0].capitalize()} closed after {span[-1]}:00. Between what times could {who} have gone to {places[0]}?")
    slots = {f"{span[k]}:00 to {span[k + 1]}:00": f"{span[k]}:00 to {span[k + 1]}:00" for k in range(len(span) - 1)}
    keys = rng.sample(list(slots), min(4, len(slots)))
    gold = f"{span[free]}:00 to {span[free + 1]}:00"
    if gold not in keys:
        keys[0] = gold
    add("temporal", i, q, "Which option is the correct answer?", {k: k for k in keys}, gold, rng)


def arithmetic(rng, i):
    def term():
        return f"({rng.randint(-9, 9)} {rng.choice(['+', '-', '*'])} {rng.randint(-9, 9)})"
    e = f"({term()} {rng.choice(['+', '-', '*'])} {term()})"
    v = eval(e)
    opts = {v}
    while len(opts) < 10:
        opts.add(v + rng.choice([-20, -10, -5, -2, -1, 1, 2, 5, 10, 20, -v * 2 if v else 3]))
    add("arithmetic", i, {"expression": e + " ="}, "Choose the numeric value of the expression in state.",
        {str(o): str(o) for o in opts}, str(v), rng)


CODE_TEMPLATES = [
    ("def f(xs):\n    return [x * {a} for x in xs if x % {b} == 0]", lambda r: [r.randint(-9, 20) for _ in range(r.randint(3, 7))]),
    ("def f(s):\n    return s[::{a}] + s[{b}:]", lambda r: "".join(r.choice("abcdefghij") for _ in range(r.randint(4, 9)))),
    ("def f(xs):\n    out = []\n    for x in xs:\n        out.append((xs.count(x), x))\n    return sorted(set(out))[-{a}:]", lambda r: [r.randint(0, 4) for _ in range(r.randint(4, 8))]),
    ("def f(s):\n    d = {{}}\n    for ch in s:\n        d[ch] = d.get(ch, 0) + {a}\n    return max(d.items(), key=lambda kv: (kv[1], kv[0]))", lambda r: "".join(r.choice("abcde") for _ in range(r.randint(4, 10)))),
    ("def f(xs):\n    total = 0\n    for i, x in enumerate(xs):\n        if i % {b} == 0:\n            total += x * {a}\n        else:\n            total -= x\n    return total", lambda r: [r.randint(-5, 9) for _ in range(r.randint(3, 8))]),
    ("def f(s):\n    return '-'.join(w[::-1] if len(w) > {b} else w.upper() for w in s.split())[:{a}0]", lambda r: " ".join("".join(r.choice("abcdefgh") for _ in range(r.randint(1, 6))) for _ in range(r.randint(2, 5)))),
]


def cruxeval(rng, i):
    tpl, gen = rng.choice(CODE_TEMPLATES)
    a, b = rng.randint(1, 3), rng.randint(2, 4)
    code = tpl.format(a=a, b=b)
    ns = {}
    exec(code, ns)
    arg = gen(rng)
    out = ns["f"](arg)
    opts = {repr(out)}
    tries = 0
    while len(opts) < 6 and tries < 50:
        tries += 1
        a2, b2 = rng.randint(1, 3), rng.randint(2, 4)
        ns2 = {}
        exec(tpl.format(a=a2, b=b2), ns2)
        try:
            opts.add(repr(ns2["f"](gen(rng) if rng.random() < 0.4 else arg)))
        except Exception:
            pass
    if len(opts) < 3:
        return
    crit = {f"option_{k}": o for k, o in enumerate(sorted(opts))}
    gold = next(k for k, o in crit.items() if o == repr(out))
    add("cruxeval", i, {"code": code, "input": repr(arg)},
        "Choose the correct output of f called with the supplied input arguments. Candidates are Python literal values.",
        crit, gold, rng)


FAMILIES = [boolean, dates, navigate, tracking, web_of_lies, ordering, temporal, arithmetic, cruxeval]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--per-family", type=int, default=1500)
    p.add_argument("--seed", type=int, default=20261002)
    a = p.parse_args()
    for fam in FAMILIES:
        rng = random.Random(f"{a.seed}:{fam.__name__}")
        for i in range(a.per_family):
            fam(rng, i)
    seen, n = set(), 0
    with open(a.out, "w") as f:
        for r in RECORDS:
            key = json.dumps([r["state"], r["questions"]["q"]["instructions"]], sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} records ({len(RECORDS) - n} duplicates dropped)")


if __name__ == "__main__":
    main()
