# Generate Path Script

This script generates a series of atomic structures along a displacement path for ferroelectric materials, specifically designed for PbTiO₃ (PTO) systems. It creates intermediate structures by gradually reducing ionic displacements, which can be useful for studying phase transitions, energy barriers, or creating animation sequences.

## Overview

The `generate-path.py` script:
1. Reads an initial atomic structure from an XYZ file
2. Calculates displacement vectors for A-site (Pb) and B-site (Ti) cations relative to their oxygen neighbors
3. Generates a series of intermediate structures by linearly interpolating between the initial displaced state and a reference state
4. Outputs numbered XYZ files for each intermediate structure

## Prerequisites

- Python 3.x
- ASE (Atomic Simulation Environment)
- FerroDispCalc package
- NumPy

## Usage

### Basic Usage

```bash
python generate-path.py
```

The script expects an input file named `PTO-T-444.xyz` in the same directory.

### Input Requirements

- **Input file**: `PTO-T-444.xyz` - An XYZ format file containing the atomic structure
- The structure should contain Pb (lead), Ti (titanium), and O (oxygen) atoms
- The script is specifically designed for PbTiO₃ (PTO) tetragonal structures

### Parameters

You can modify the following parameters in the script:

- `nimage = 15` - Number of intermediate structures to generate (default: 15)
- Neighbor list parameters:
  - Ti-O neighbors: 4 minimum, 6 maximum neighbors
  - Pb-O neighbors: 5 minimum, 12 maximum neighbors

### Output

The script generates:
- **XYZ files**: `01.xyz`, `02.xyz`, ..., `15.xyz` (numbered sequentially)
- **Console output**: For each structure, prints the image number and mean displacement values

Example console output:
```
0 -0.123456 -0.234567
1 -0.118743 -0.224123
2 -0.114030 -0.213679
...
```

## How It Works

1. **Initial Setup**: Reads the input structure and builds neighbor lists for Pb-O and Ti-O pairs
2. **Displacement Calculation**: Computes initial displacement vectors using FerroDispCalc
3. **Path Generation**: Creates linear interpolation between initial displacements and zero displacement
4. **Structure Generation**: For each point along the path:
   - Copies the original structure
   - Applies the interpolated displacements
   - Calculates new displacement values
   - Saves the modified structure

## Technical Details

### Neighbor Lists
- **Pb-O (A-site)**: 5-12 neighbors typically for the larger Pb²⁺ cation
- **Ti-O (B-site)**: 4-6 neighbors typically for the smaller Ti⁴⁺ cation

### Displacement Calculation
The script uses FerroDispCalc's `get_displacement()` method to calculate how atoms are displaced from their ideal positions relative to their oxygen coordination environment.

### Linear Interpolation
The path uses `numpy.linspace()` to create a linear interpolation between:
- **Start**: Current displaced positions
- **End**: Zero displacement (reference state)

## Example Workflow

1. Prepare your PTO structure file as `PTO-T-444.xyz`
2. Run the script: `python generate-path.py`
3. The script will generate 15 intermediate structures (01.xyz through 15.xyz)
4. Use these structures for further analysis, visualization, or energy calculations

## Customization

To modify the script for different systems or parameters:

```python
# Change number of images
nimage = 20  # Generate 20 intermediate structures

# Modify neighbor list parameters
nl_bo = NeighborList(atoms).build(["Ti"], ["O"], 3, 7)  # Different Ti-O neighbors
nl_ao = NeighborList(atoms).build(["Pb"], ["O"], 4, 10) # Different Pb-O neighbors

# Change input file
atoms = read("your_structure.xyz")
```

## Applications

- **Phase transition studies**: Track atomic movements during ferroelectric phase transitions
- **Energy barrier calculations**: Generate structures for nudged elastic band (NEB) calculations
- **Visualization**: Create smooth animation sequences of atomic movements
- **Structure analysis**: Study how displacement patterns evolve

## Notes

- The script assumes a specific naming convention for output files (01.xyz, 02.xyz, etc.)
- Displacement calculations are performed in Cartesian coordinates
- The linear interpolation may not represent the actual minimum energy path between states