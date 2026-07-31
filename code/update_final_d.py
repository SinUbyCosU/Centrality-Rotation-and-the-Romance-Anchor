import re

with open('C:\\Users\\Tanushree\\.gemini\\antigravity-ide\\brain\\3938ca46-534e-44d5-b9f3-e957cc8dfbd5\\final_paper_tables.md', 'r', encoding='utf-8') as f:
    text = f.read()

# I will find Table D and remove qwen-7b
lines = text.split('\n')
new_lines = []
for line in lines:
    if line.startswith('| qwen-7b |'):
        continue
    new_lines.append(line)

with open('C:\\Users\\Tanushree\\.gemini\\antigravity-ide\\brain\\3938ca46-534e-44d5-b9f3-e957cc8dfbd5\\final_paper_tables.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(new_lines))
