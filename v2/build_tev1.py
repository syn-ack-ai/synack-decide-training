"""Convert tev1's new-v1 records (state/question/options) into v2 decision records."""
import argparse
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--records", required=True, help="tev1 data/new-v1/records/{train,dev}.jsonl")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    n = 0
    with open(a.records) as f, open(a.out, "w") as g:
        for line in f:
            r = json.loads(line)
            crit = {o["key"]: o["description"] for o in r["options"]}
            g.write(json.dumps({"id": f"tev1:{r['id']}", "source": f"tev1/{r['source']}", "state": r["state"],
                                "questions": {"q": {"type": "choice", "instructions": r["question"], "criteria": crit}},
                                "expected": {"q": r["answer_key"]}}, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n}")


if __name__ == "__main__":
    main()
