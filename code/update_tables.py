import re

with open('C:\\Users\\Tanushree\\.gemini\\antigravity-ide\\brain\\3938ca46-534e-44d5-b9f3-e957cc8dfbd5\\final_paper_tables.md', 'r', encoding='utf-8') as f:
    text = f.read()

# I want to drop Table E and Table F
# Table E starts with "## Table E" and goes to end
text = re.split(r'## Table E', text)[0]

# Now read the generated tables and replace them
with open('c:\\Users\\Tanushree\\Downloads\\work\\tableB_fixed.md', 'r', encoding='utf-8', errors='replace') as f:
    table_b = f.read().replace('A\ufffd', '°')

with open('c:\\Users\\Tanushree\\Downloads\\work\\tableD.md', 'r', encoding='utf-8') as f:
    table_d = f.read()

# Replace Table B
b_start = text.find('## Table B')
c_start = text.find('## Table C')
text = text[:b_start] + "## Table B — SCD × Robustness (Cohen's d, corrected, 16 models)\n" + table_b + "\n" + text[c_start:]

# Replace Table D
d_start = text.find('## Table D')
text = text[:d_start] + "## Table D — Moral Foundations Alignment (Phase 17)\n" + table_d + "\n"

with open('C:\\Users\\Tanushree\\.gemini\\antigravity-ide\\brain\\3938ca46-534e-44d5-b9f3-e957cc8dfbd5\\final_paper_tables.md', 'w', encoding='utf-8') as f:
    f.write(text)
