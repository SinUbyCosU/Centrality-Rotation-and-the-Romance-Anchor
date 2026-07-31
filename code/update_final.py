import re

with open('C:\\Users\\Tanushree\\.gemini\\antigravity-ide\\brain\\3938ca46-534e-44d5-b9f3-e957cc8dfbd5\\final_paper_tables.md', 'r', encoding='utf-8') as f:
    text = f.read()

with open('c:\\Users\\Tanushree\\Downloads\\work\\tableB_updated.md', 'r', encoding='utf-8') as f:
    table_b = f.read()

with open('c:\\Users\\Tanushree\\Downloads\\work\\tableC_updated.md', 'r', encoding='utf-8') as f:
    table_c = f.read()

# Replace Table B
b_start = text.find("## Table B")
c_start = text.find("## Table C")
d_start = text.find("## Table D")

new_text = text[:b_start] + "## Table B — SCD × Robustness (Cohen's d, corrected, 15 models)\n" + table_b + "\n\n" + \
           "## Table C — SCD × Adversarial Success (Phase 11, m=13, FDR-corrected)\n" + table_c + "\n\n" + \
           text[d_start:]

with open('C:\\Users\\Tanushree\\.gemini\\antigravity-ide\\brain\\3938ca46-534e-44d5-b9f3-e957cc8dfbd5\\final_paper_tables.md', 'w', encoding='utf-8') as f:
    f.write(new_text)
