import re

with open('/Users/jinho/Jeju_predict/scratch/rendered.html', 'r', encoding='utf-8') as f:
    html = f.read()

# 주석 제거
html_clean = re.sub(r'<!--.*?-->', '', html, flags=re.DOTALL)

# 단순 tag parser
tags = re.findall(r'</?([a-zA-Z0-9\-]+)(?:\s+[^>]*?)?>', html_clean)

stack = []
for i, tag in enumerate(re.finditer(r'(</?([a-zA-Z0-9\-]+)(?:\s+[^>]*?)?>)', html_clean)):
    tag_str = tag.group(1)
    tag_name = tag.group(2).lower()
    is_closing = tag_str.startswith('</')
    
    # self closing tags
    if tag_str.endswith('/>') or tag_name in ['img', 'br', 'hr', 'input', 'meta', 'link']:
        continue
        
    if not is_closing:
        # tag 속성에서 id 파싱
        id_match = re.search(r'id=["\']([^"\']+)["\']', tag_str)
        tag_id = id_match.group(1) if id_match else None
        
        stack.append((tag_name, tag_id, tag_str))
    else:
        if stack:
            popped_name, popped_id, popped_str = stack.pop()
            if popped_name != tag_name:
                print(f"Mismatch: Opened {popped_name} (id={popped_id}) but closed with {tag_name}")
        else:
            print(f"Extra closing tag: {tag_str}")

print("Remaining in stack:")
for s in stack:
    print(f"Opened: {s[0]} id={s[1]}")
