import json, os, glob, datetime
base = r"F:\CODE\New\baseline\data"
for name in ["eval_multi_6ab_chain", "eval_multi_6ab_chain2", "eval_multi_6ab_zone2",
             "eval_phased_multi_chain", "eval_phased_multi_chain2", "eval_phased_multi_zone2",
             "eval_phased_single", "eval_phased_single_bias", "eval_multi_6ab_e3b_scan"]:
    p = os.path.join(base, name, "summary.json")
    if os.path.exists(p):
        d = json.load(open(p, encoding="utf-8"))
        print(name, "->", json.dumps(d, ensure_ascii=False)[:220])
print("\n--- latest eval dirs ---")
dirs = [d for d in glob.glob(os.path.join(base, "eval_*")) if os.path.isdir(d)]
dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
for d in dirs[:10]:
    print(" ", os.path.basename(d), datetime.datetime.fromtimestamp(os.path.getmtime(d)).strftime("%m-%d %H:%M"))
