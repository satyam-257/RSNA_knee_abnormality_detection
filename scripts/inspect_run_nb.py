import json

with open("outputs/dinosaurs_ensemble/run_nb/rsna-knee-speedy-coatnet-dinosaurs-ensemble.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        outs = c.get("outputs", [])
        errs = [o for o in outs if o.get("output_type") == "error"]
        texts = [o.get("text", "") for o in outs if "text" in o]
        all_text = "".join("".join(t) if isinstance(t, list) else t for t in texts)
        first_line = all_text.strip().split("\n")[0] if all_text.strip() else ""
        last_line = all_text.strip().split("\n")[-1] if all_text.strip() else ""
        if errs:
            print(f"Cell {i:02d}: ERROR {errs[0].get('ename')}: {errs[0].get('evalue')}")
        else:
            print(f"Cell {i:02d}: {first_line[:50]} ... {last_line[-50:]}")
