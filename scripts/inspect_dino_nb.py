import json

with open("scratch_notebook/bend-the-knee-to-the-dinosaurs.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, cell in enumerate(nb["cells"]):
    ct = cell["cell_type"]
    src = "".join(cell["source"]).strip()
    first_line = src.split("\n")[0] if src else ""
    if ct == "markdown":
        print(f"Cell {i:02d} [MD  ]: {first_line[:90]}")
    else:
        lines = [l for l in src.split("\n") if l.strip() and not l.strip().startswith("#")]
        code_peek = lines[0][:90] if lines else "empty"
        print(f"Cell {i:02d} [CODE]: {code_peek}")
