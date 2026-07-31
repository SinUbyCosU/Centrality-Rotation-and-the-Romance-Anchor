import json

text = open('tinyllama_judged.jsonl', encoding='utf-8').read()

decoder = json.JSONDecoder()
pos = 0
complied = 0
refused = 0

while pos < len(text):
    text_slice = text[pos:].lstrip()
    if not text_slice:
        break
    try:
        obj, index = decoder.raw_decode(text_slice)
        # We wrote 'llm_classification' as the judge output
        llm_class = obj.get('llm_classification', '')
        if llm_class == 'refused':
            refused += 1
        elif llm_class == 'complied':
            complied += 1
        else:
            # Fallback
            if obj.get('is_safe', False):
                refused += 1
            else:
                complied += 1
                
        pos += len(text[pos:]) - len(text_slice) + index
    except json.JSONDecodeError as e:
        # If there's an explicit literal \n string causing issues, advance past it
        if text_slice.startswith('\\n'):
            pos += len(text[pos:]) - len(text_slice) + 2
        else:
            print('Error at', e)
            break

print('TOTAL:', complied + refused)
print('COMPLIED:', complied)
print('REFUSED (SAFE):', refused)
print('ASR (Attack Success Rate): {:.2f}%'.format(complied / (complied + refused) * 100))
