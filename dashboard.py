#!/usr/bin/env python3
"""
dashboard.py — CG-CRASH 论文进度看板
Usage: python3 dashboard.py  # 生成 output/dashboard.html
"""
import json
from pathlib import Path
from datetime import datetime

ROOT = Path('/data/sony/LFCRASH/LFCRASH-CBM')

def load(p):
    try:
        return json.load(open(p))
    except:
        return None

def tail(p, n=6):
    try:
        lines = open(p).readlines()
        return ''.join(lines[-n:])
    except:
        return 'N/A'

def main():
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Load results
    crash = load(ROOT/'output'/'v3_final'/'crash_full'/'results.json') or {}
    a3d   = load(ROOT/'output'/'v3_final'/'a3d_full'/'results.json') or {}
    dad   = load(ROOT/'output'/'dad_sota_push'/'dad_z512'/'results.json') or {}

    # Ablation
    abl_datasets = ['crash','a3d','dad']
    abl_keys = ['full','no_cbm','no_align','no_sparse','no_recon']
    abl_labels = {'full':'Full','no_cbm':'w/o CBM','no_align':'w/o Align',
                  'no_sparse':'w/o Sparse','no_recon':'w/o Recon'}
    abl_data = {}
    for ds in abl_datasets:
        abl_data[ds] = {}
        for k in abl_keys:
            r = load(ROOT/'output'/'v3_final'/f'{ds}_{k}'/'results.json')
            abl_data[ds][k] = r or {}

    # Training logs
    ft_log  = tail(ROOT/'output'/'dad_finetune_z256.log', 8)
    cur_log = tail(ROOT/'output'/'dad_curriculum_v1.log', 5)
    ns_log  = tail(ROOT/'output'/'dad_no_sparse_long.log', 5)

    html = f"""
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="60">
<title>CG-CRASH Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Noto+Sans+SC:wght@400;700&display=swap');
  :root {{
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --green: #3fb950; --blue: #58a6ff; --orange: #f0883e;
    --red: #f85149; --text: #e6edf3; --muted: #8b949e;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Noto Sans SC', sans-serif; padding: 24px; }}
  h1 {{ font-size: 1.6rem; color: var(--blue); margin-bottom: 4px; }}
  .ts {{ color: var(--muted); font-size: .85rem; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; margin-bottom: 24px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }}
  .card h3 {{ font-size: .9rem; color: var(--muted); margin-bottom: 12px; text-transform: uppercase; letter-spacing: .05em; }}
  .metric {{ font-size: 2rem; font-weight: 700; color: var(--green); font-family: 'JetBrains Mono', monospace; }}
  .sub {{ font-size: .85rem; color: var(--muted); margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: .82rem; font-family: 'JetBrains Mono', monospace; }}
  th {{ color: var(--muted); padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--border); }}
  th:first-child {{ text-align: left; }}
  td {{ padding: 6px 10px; text-align: right; border-bottom: 1px solid #21262d; }}
  td:first-child {{ text-align: left; color: var(--blue); }}
  .best {{ color: var(--green); font-weight: 700; }}
  .log {{ background: #010409; border: 1px solid var(--border); border-radius: 6px;
          padding: 12px; font-family: 'JetBrains Mono', monospace; font-size: .75rem;
          color: #adbac7; white-space: pre-wrap; word-break: break-all; margin-top: 8px; }}
  .section {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
              padding: 20px; margin-bottom: 16px; }}
  .section h2 {{ font-size: 1rem; color: var(--blue); margin-bottom: 16px; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
          font-size: .75rem; font-family: 'JetBrains Mono', monospace;
          background: #1f6feb33; color: var(--blue); border: 1px solid #1f6feb66; }}
</style>
</head>
<body>
<h1>CG-CRASH — 论文进度看板</h1>
<p class="ts">最后更新: {now} (每60秒自动刷新)</p>

<div class="grid">
  <div class="card">
    <h3>CRASH AP <span class="tag">v3_final</span></h3>
    <div class="metric">{crash.get('AP',0)*100:.2f}%</div>
    <div class="sub">mTTA={crash.get('mTTA',0):.3f}s &nbsp;|&nbsp; TTA@R80={crash.get('TTA_R80',0):.3f}s</div>
    <div class="sub">vs CVPR'22: +{(crash.get('AP',0)-0.9739)*100:.2f}% AP</div>
  </div>
  <div class="card">
    <h3>A3D AP <span class="tag">v3_final</span></h3>
    <div class="metric">{a3d.get('AP',0)*100:.2f}%</div>
    <div class="sub">mTTA={a3d.get('mTTA',0):.3f}s &nbsp;|&nbsp; TTA@R80={a3d.get('TTA_R80',0):.3f}s</div>
    <div class="sub">vs UniVAD'23: +{(a3d.get('AP',0)-0.9150)*100:.2f}% AP &#x1F525;</div>
  </div>
  <div class="card">
    <h3>DAD AP <span class="tag">dad_z512</span></h3>
    <div class="metric">{dad.get('AP',0)*100:.2f}%</div>
    <div class="sub">mTTA={dad.get('mTTA',0):.3f}s &nbsp;|&nbsp; TTA@R80={dad.get('TTA_R80',0):.3f}s</div>
    <div class="sub">Training in progress → target >66%</div>
  </div>
</div>

<div class="section">
  <h2>Ablation Study (v3_final)</h2>
  <table>
    <tr><th>Condition</th>
      <th>CRASH AP</th><th>mTTA</th>
      <th>A3D AP</th><th>mTTA</th>
      <th>DAD AP</th><th>mTTA</th></tr>"""

    best_crash_ap = max((abl_data['crash'][k].get('AP',0) for k in abl_keys), default=0)
    best_a3d_ap   = max((abl_data['a3d'][k].get('AP',0)   for k in abl_keys), default=0)
    best_dad_ap   = max((abl_data['dad'][k].get('AP',0)   for k in abl_keys), default=0)

    for k in abl_keys:
        cr = abl_data['crash'][k]; ar = abl_data['a3d'][k]; dr = abl_data['dad'][k]
        cap = cr.get('AP',0); aap = ar.get('AP',0); dap = dr.get('AP',0)
        cc = ' class="best"' if abs(cap-best_crash_ap)<0.0005 else ''
        ac = ' class="best"' if abs(aap-best_a3d_ap)<0.0005 else ''
        dc = ' class="best"' if abs(dap-best_dad_ap)<0.0005 else ''
        html += f"""
    <tr><td>{abl_labels[k]}</td>
      <td{cc}>{cap*100:.2f}%</td><td>{cr.get('mTTA',0):.3f}s</td>
      <td{ac}>{aap*100:.2f}%</td><td>{ar.get('mTTA',0):.3f}s</td>
      <td{dc}>{dap*100:.2f}%</td><td>{dr.get('mTTA',0):.3f}s</td></tr>"""

    html += """
  </table>
</div>

<div class="section">
  <h2>Active Training Jobs</h2>
  <h3 style="color:var(--orange);font-size:.85rem;margin-bottom:6px">▶ dad_finetune_z256 (GPU4)</h3>
  <div class="log">""" + ft_log.replace('<','&lt;').replace('>','&gt;') + """</div>
  <h3 style="color:var(--orange);font-size:.85rem;margin:12px 0 6px">▶ dad_curriculum_v1 (GPU1)</h3>
  <div class="log">""" + cur_log.replace('<','&lt;').replace('>','&gt;') + """</div>
  <h3 style="color:var(--orange);font-size:.85rem;margin:12px 0 6px">▶ dad_no_sparse_long (GPU2, λ_sparse=0)</h3>
  <div class="log">""" + ns_log.replace('<','&lt;').replace('>','&gt;') + """</div>
</div>

</body></html>"""

    out = ROOT / 'output' / 'dashboard.html'
    out.write_text(html)
    print(f'Dashboard written to {out}')

if __name__ == '__main__':
    main()
