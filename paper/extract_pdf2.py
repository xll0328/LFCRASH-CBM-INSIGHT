from pdfminer.high_level import extract_text
text = extract_text('/data/sony/LFCRASH/LFCRASH-CBM/paper/iclr26.pdf', maxpages=8)
with open('/data/sony/LFCRASH/LFCRASH-CBM/paper/extracted/iclr26.txt','w') as f:
    f.write(text)
print(f'Chars: {len(text)}')
print(text[:3000])
