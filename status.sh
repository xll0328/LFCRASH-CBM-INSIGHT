#!/bin/bash
# status.sh — CG-CRASH 项目一键状态查看
echo ''
echo '======================================================'
echo '  CG-CRASH Project Status'
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo '======================================================'

echo ''
echo '--- Training Jobs ---'
for pid in $(pgrep -f 'train_dad_curriculum' 2>/dev/null); do
    cmd=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
    if echo "$cmd" | grep -q '\-\-gpu'; then
        cpu=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ')
        tag=$(echo "$cmd" | grep -o '\-\-tag [^ ]*' | awk '{print $2}')
        gpu=$(echo "$cmd" | grep -o '\-\-gpu [^ ]*' | awk '{print $2}')
        echo "  PID=$pid GPU=$gpu tag=$tag CPU=${cpu}%"
    fi
done

echo ''
echo '--- GPU Status ---'
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader | awk -F',' \
  '{printf "  GPU%s: %s used, %s free, util=%s\n",$1,$2,$3,$4}'

echo ''
echo '--- DAD Curriculum (v1) ---'
grep -E 'Epoch|EVAL|\*\*\*' /data/sony/LFCRASH/LFCRASH-CBM/output/dad_curriculum_v1.log 2>/dev/null | tail -5 | awk '{print "  "$0}'

echo ''
echo '--- DAD Finetune (z256) ---'
grep -E 'Epoch|EVAL|\*\*\*' /data/sony/LFCRASH/LFCRASH-CBM/output/dad_finetune_z256.log 2>/dev/null | tail -6 | awk '{print "  "$0}'

echo ''
echo '--- Best Results (v3_final) ---'
echo -n '  CRASH: '
python3 -c "
import json
d=json.load(open('/data/sony/LFCRASH/LFCRASH-CBM/output/v3_final/crash_full/results.json'))
print(f'AP={d[\"AP\"]*100:.2f}% mTTA={d[\"mTTA\"]:.3f}s TTA@R80={d[\"TTA_R80\"]:.3f}s')
" 2>/dev/null || echo 'N/A'
echo -n '  A3D:   '
python3 -c "
import json
d=json.load(open('/data/sony/LFCRASH/LFCRASH-CBM/output/v3_final/a3d_full/results.json'))
print(f'AP={d[\"AP\"]*100:.2f}% mTTA={d[\"mTTA\"]:.3f}s TTA@R80={d[\"TTA_R80\"]:.3f}s')
" 2>/dev/null || echo 'N/A'
echo -n '  DAD:   '
python3 -c "
import json
candidates = [
    '/data/sony/LFCRASH/LFCRASH-CBM/output/v3_final/dad_no_sparse/results.json',
    '/data/sony/LFCRASH/LFCRASH-CBM/output/dad_sota_push/dad_z512/results.json',
    '/data/sony/LFCRASH/LFCRASH-CBM/output/dad_best_final/results.json',
]
best = None
for p in candidates:
    try:
        d = json.load(open(p))
        if best is None or d.get('AP',0) > best.get('AP',0):
            best = d; best['_src'] = p.split('/')[-2]
    except: pass
if best:
    print(f'AP={best[\"AP\"]*100:.2f}% mTTA={best[\"mTTA\"]:.3f}s TTA@R80={best[\"TTA_R80\"]:.3f}s ({best[\"_src\"]})')
" 2>/dev/null || echo 'N/A'

echo ''
echo '--- Paper Figures (paper_figures_v2/) ---'
ls /data/sony/LFCRASH/LFCRASH-CBM/output/paper_figures_v2/*.pdf 2>/dev/null | \
  while read f; do
    size=$(du -h "$f" | cut -f1)
    echo "  $size  $(basename $f)"
  done

echo ''
echo '======================================================'
