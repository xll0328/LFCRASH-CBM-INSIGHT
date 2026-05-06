from pdfminer.high_level import extract_text
text = extract_text('/data/sony/LFCRASH/LFCRASH-CBM/paper/iclr26.pdf', maxpages=20)
with open('/data/sony/LFCRASH/LFCRASH-CBM/paper/extracted/iclr26.txt','w') as f:
    f.write(text)
print(f'Chars: {len(text)}')
# Find key results sections
import re
for kw in ['DAD','A3D','CCD','Table','AP','mTTA','91','89','77','68','96','SOTA']:
    for m in re.finditer(kw, text):
        idx = m.start()
        snippet = text[max(0,idx-50):idx+150].replace('\n',' ')
        if any(c.isdigit() for c in snippet):
            print(f'[{kw}@{idx}] {snippet}')
            print()
            break
