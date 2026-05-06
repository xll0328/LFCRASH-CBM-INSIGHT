from pypdf import PdfReader
r = PdfReader('/data/sony/LFCRASH/LFCRASH-CBM/paper/iclr26.pdf')
text = ''
for p in r.pages:
    text += p.extract_text() + '\n'
with open('/data/sony/LFCRASH/LFCRASH-CBM/paper/extracted/iclr26.txt','w') as f:
    f.write(text)
print(f'Pages: {len(r.pages)}, Chars: {len(text)}')
