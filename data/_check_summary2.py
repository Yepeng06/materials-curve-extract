import json, os
base = r"F:\CODE\New\baseline\data"
for name in ["eval_official6a_chain", "eval_official6a_phased", "eval_recall_tau_chain", "eval_e3b_fullscan", "eval_final_scan"]:
    p = os.path.join(base, name, "summary.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        print(name, "->", json.dumps(d, ensure_ascii=False)[:260])
