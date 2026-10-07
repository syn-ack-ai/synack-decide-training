import json, sys, collections, torch
sys.path.insert(0, "v2")
from engine_v2 import load_v2
from common import LABELS, SYSTEM
rows = [json.loads(l) for l in open("v2/data/mix_valid.jsonl")][:400]
for name, adapter in [("base", None), ("pilot", "v2/adapters/pilot")]:
    tok, m, lid = load_v2("models/gemma4-12b-it", adapter, True)
    pos, ok, per = collections.Counter(), 0, collections.defaultdict(lambda: [0, 0])
    for e in rows:
        ids = tok.apply_chat_template(e["messages"][:-1], add_generation_prompt=True, enable_thinking=False, tokenize=True, return_dict=False)
        with torch.inference_mode():
            lg = m(input_ids=torch.tensor([ids], device="cuda"), logits_to_keep=1).logits[0, -1]
        pred = LABELS[int(lg[lid[:len(e["labels"])]].argmax())]
        pos[pred] += 1; c = pred == e["gold"]; ok += c
        s = e["source"].split("/")[1] if e["source"].startswith("hf/") else e["source"].split("/")[0]
        per[s][0] += c; per[s][1] += 1
    gold = collections.Counter(e["gold"] for e in rows)
    print(f"== {name}: acc {ok}/{len(rows)}  pred-pos {dict(pos.most_common(5))}  gold-pos {dict(gold.most_common(5))}")
    print("   ", {k: f"{a}/{b}" for k, (a, b) in sorted(per.items())})
    del m; torch.cuda.empty_cache()
