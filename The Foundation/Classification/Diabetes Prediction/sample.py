import json
import nbformat as nbf

# --- 1. Load your JSON file ---
with open("sample.json", "r", encoding="utf-8") as f:
    raw_json = json.load(f)

# --- 2. Create a new notebook structure ---
nb = nbf.v4.new_notebook()

# --- 3. Copy cells into proper Jupyter format ---
cells = []
for cell in raw_json.get("cells", []):
    cell_type = cell.get("cell_type", "code")
    source = cell.get("source", [])
    if isinstance(source, list):
        source = "".join(source)  # merge list of strings into one string

    if cell_type == "markdown":
        new_cell = nbf.v4.new_markdown_cell(source)
    elif cell_type == "code":
        new_cell = nbf.v4.new_code_cell(source)
    else:
        continue  # skip unsupported cell types

    cells.append(new_cell)

nb["cells"] = cells

# --- 4. Add basic metadata (kernel info, nbformat version, etc.) ---
nb["metadata"] = {
    "kernelspec": {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python"
    },
    "language_info": {
        "name": "python",
        "version": "3.x"
    }
}

# --- 5. Save as .ipynb file ---
output_path = "sample_converted.ipynb"
with open(output_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ Converted JSON → {output_path}")
