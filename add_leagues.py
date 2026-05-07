import json
import re

notebook_path = "data3_advanced copy.ipynb"
file1_path = "data3_advanced.py"

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open(file1_path, 'r', encoding='utf-8') as f:
    content = f.read()

leagues = [
    ("Premier League", "E0"),
    ("La Liga", "SP1"),
    ("Serie A", "I1"),
    ("Bundesliga", "D1"),
    ("Ligue 1", "F1")
]

for name, code in leagues:
    replacement = f"""scenarios = [
    {{"name": "{name}", "filter": ['{code}']}}
]"""
    modified_content = re.sub(r"scenarios\s*=\s*\[.*?\]", replacement, content, flags=re.DOTALL)
    
    nb['cells'].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"## {name} Pipeline\n"]
    })
    
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [modified_content]
    })

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Added league cells.")
