import json, os
from collections import Counter

d = json.load(open('data/eval_multi_diag_goi/diag.json', encoding='utf-8'))
axis = Counter()
per_axis = Counter()
for r in d['rows']:
    if r.get('n_gt', 0) <= 0:
        continue
    name = r['image']
    stem = ('data/val_multi/' + name[:-4]) if os.path.exists('data/val_multi/' + name) else ('data/val_single/' + name[:-4])
    mp = stem + '_meta.json'
    if not os.path.exists(mp):
        continue
    m = json.load(open(mp, encoding='utf-8'))
    xk = m.get('x_kind', '?')
    yk = m.get('y_kind', '?')
    for g in r['per_gt']:
        rel = g['rel']
        if 0.01 < rel <= 0.02:
            axis[(xk, yk)] += 1
        elif rel > 0.02:
            per_axis[(xk, yk)] += 1
print('1-2% bucket by axis kinds:', dict(axis))
print('>2% bucket by axis kinds:', dict(per_axis))
print('1-2% total:', sum(axis.values()), ' >2% total:', sum(per_axis.values()))
# log involvement
log_any = sum(v for (xk, yk), v in axis.items() if 'log' in (xk, yk))
print('1-2% curves on log axes (x or y):', log_any)
