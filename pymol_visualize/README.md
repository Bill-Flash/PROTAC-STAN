
# PyMOL Visualization of Key Residues

## Introduction

An example script for visualizing key residues on protein structures using PyMOL, to assist with structural analysis and functional studies.

## Installation

Make sure PyMOL is installed. You can install it with the following command:

```bash
conda install -c schrodinger -c conda-forge pymol
```

## Usage

Activate your environment and run the script:

```bash
conda activate pymol
```

In the GUI, run the following command:

```bash
run visualize_by_pymol.py
```

## Complete Usage Example:

Let's take the protein with PDB ID `7KHH` as an example and demonstrate step by step how to use the script for visualization.

### Prepare Input Files

- PDB file: `7khh_prep.pdb`
- CSV file: `residue_data.csv`

The PDB file contains the protein structure to be visualized, and the CSV file includes information about the residues of interest.

### Launch PyMOL

Enter the following command in the terminal:

```bash
pymol
```

### Automatic Visualization

Once the input files are ready, enter the automation script command in the PyMOL command line:

```bash
run visualize_by_pymol.py
```

### Manual Visualization

We can also manually execute the commands from the script step by step to achieve visualization. Here are the commands to follow:

1. Load the PDB file:
    ```bash
    load 7khh_prep.pdb, prot
    ```
2. Setup:
    ```bash
    bg_color white
    hide everything, prot
    show cartoon, prot
    set cartoon_transparency, 0.8
    ```
3. Highlight key residues: Manually select and visualize residues of interest. For example, to highlight residue 100 in chain C:
    ```bash
    sele = "chain C and resi 100"
    sele_name = "residue_to_highlight"
    color = "green"
    select(sele_name, sele)
    show("sticks", sele_name)
    color(color, sele_name)
    set("stick_radius", 0.5, sele_name)
    label(sele+" and name CA", "resn + '-' + resi") 
    ```
4. Customize visualization settings:
    ```bash
    set("stick_color", "atomic")  # Color sticks according to coloring
    set("label_size", 16)         # Label font size
    set("float_labels", 1)        # Labels not occluded
    set("valence", 1)             # Show valence
    ```

### Example output:

<img src="example.jpg" alt="PyMOL Visualization Example" width="600"/>