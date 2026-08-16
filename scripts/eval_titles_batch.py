import csv, glob, json, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
from mci.pipeline.extractor import Extractor

def norm(s):
    return ''.join(str(s).lower().split()).replace('-', '').replace('.', '')

ex = Extractor(ocr_backend='paddle', segmenter='unet')
imgs = sorted(p for p in glob.glob('data/eval_b2/*.png') if not p.endswith('_mask.png'))
rows = []
for img in imgs:
    stem = img[:-4]
    meta = json.load(open(stem + '_meta.json', encoding='utf-8'))
    row = {'image': os.path.basename(img)}
    try:
        r = ex.extract(img)
        titles = r.meta.get('titles', {})
        row['status'] = 'ok'
        for key in ('title', 'x_label', 'y_label'):
            t = titles.get(key) or {}
            row[key] = '1' if t.get('text') else '0'
        xt = titles.get('x_label') or {}
        yt = titles.get('y_label') or {}
        row['x_var'] = '1' if norm(xt.get('variable', '')) == norm(meta.get('x_label', '')) else '0'
        row['y_var'] = '1' if norm(yt.get('variable', '')) == norm(meta.get('y_label', '')) else '0'
        row['x_unit'] = '1' if norm(xt.get('unit', '')) == norm(meta.get('x_unit', '')) else '0'
        row['y_unit'] = '1' if norm(yt.get('unit', '')) == norm(meta.get('y_unit', '')) else '0'
        print('[OK]', row['image'], {k: (titles.get(k) or {}).get('text', '') for k in ('title', 'x_label', 'y_label')})
    except Exception as e:
        row['status'] = 'failed'
        row['error'] = str(e)[:120]
        print('[FAIL]', row['image'], e)
    rows.append(row)

os.makedirs('data/eval_b2_out', exist_ok=True)
with open('data/eval_b2_out/report.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader()
    w.writerows(rows)
print('report written')