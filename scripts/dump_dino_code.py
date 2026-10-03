import json

with open("scratch_notebook/bend-the-knee-to-the-dinosaurs.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("scratch_notebook/full_notebook_code.py", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb["cells"]):
        ct = cell["cell_type"]
        src = "".join(cell["source"])
        out.write(f"\n{'='*80}\n# CELL {i} ({ct})\n{'='*80}\n")
        if ct == "markdown":
            for line in src.split("\n"):
                out.write(f"# [MD] {line}\n")
        else:
            out.write(src)
            out.write("\n")

print("Dumped to scratch_notebook/full_notebook_code.py")
