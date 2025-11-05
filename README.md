# ferrodispcalc

Utilities for preprocessing and postprocessing molecular simulations of ferroelectric materials.

## Purpose
-------
`ferrodispcalc` provides small, focused tools to compute ionic displacements, polarization and octahedral tilts for perovskite-like materials, and to visualize results efficiently.

## Highlights
----------
- Build neighbor list with easy-to-use API.
- Compute ionic displacements from trajectories / single structure.
- Compute polarization and octahedral-tilt descriptors for perovskite-type structures.
- On-demand visualization helpers for common plots.
- LAMMPS plugin for efficient, in-situ calculations during MD runs.

## Gallery
-------
### The local polarization of (PbTiO<sub>3</sub>)/(SrTiO<sub>3</sub>) superlattice in real space.

<div align=center>  
<img src="./gallery/PTO-STO-superlattice/profile/YZ_0.png" width=40%>
</div>

__Path:__ `./gallery/PTO-STO-superlattice/plot.py` 

### Profile of Head-to-head charged 180-degree domain wall in PbTiO<sub>3</sub>

<div align=center>  
<img src="./gallery/PTO-HH-DW/PTO-HH-DW-profile.png" width=100%>
</div>

__Path:__ `./gallery/PTO-HH-DW/plot.py` 


## Documentation
-------------
More detailed information, example usage, and API docs: 
https://moseyqaq.github.io/ferrodispcalc/

## Install
-------
```bash
git clone https://github.com/MoseyQAQ/ferrodispcalc.git
cd ferrodispcalc
pip install -e .
```

We plan to publish this package to PyPI soon.

## Credit

Please cite this repo if it helps your research. Thanks!
