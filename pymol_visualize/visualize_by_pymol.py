from pymol import cmd
import csv

# === Set input paths ===
pdb_file = "pdb/7khh_prep.pdb"
csv_file = "residue_data.csv"

# === Load structure and set background ===
cmd.load(pdb_file, "prot")
cmd.bg_color("white")
cmd.hide("everything", "prot")
cmd.show("cartoon", "prot")
cmd.set("cartoon_transparency", 0.8)

# === Define color mapping ===
role_colors = {
    "weighted": "grey90",
    "interacting": "green",
    "interacting+weighted": "magenta"
}

# === Selection set for highlighted residues ===
highlighted_sele = []

with open(csv_file, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        chain = row["chain"]
        resi = row["resi"]
        role = row["role"]
        print(f"Processing residue: {chain}:{resi} Role: {role}")
        color = role_colors.get(role)

        if not color:
            print(f"⚠️ Unrecognized role: {role}")
            continue

        # Create precise selection
        sele = f"(chain {chain} and resi {resi})"
        sele_name = f"{chain}_{resi}"
        cmd.select(sele_name, sele)

        # Color and show as sticks for emphasis
        cmd.show("sticks", sele_name)
        cmd.color(color, sele_name)
        cmd.set("stick_radius", 0.5, sele_name)  # More prominent

        # Add label: show chain+resi
        cmd.label(sele+" and name CA", "resn + '-' + resi")

        highlighted_sele.append(sele_name)

# === Zoom to highlighted residues ===
if highlighted_sele:
    cmd.zoom(" or ".join(highlighted_sele))

cmd.set("stick_color", "atomic")  # Color sticks according to coloring
cmd.set("label_size", 16)         # Label font size
cmd.set("float_labels", 1)        # Labels not occluded
cmd.set("valence", 1)             # Show valence bonds