import sys 
import os 
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, base_dir)
from ferrodispcalc.build_neighbor_list import NeighborListABO3
from ferrodispcalc import type_map
nl = NeighborListABO3(file_name='traj.lammpstrj',
                      type_map=type_map.UniPero,
                      format='lmp-dump')

nl.build(center_elements=['Zr'],
         neighbor_elements=['O'],
         cutoff=3.5,
         neighbor_num=6,
         defect=False)

nl.write('B.dat')

nl.build(center_elements=['Pb','Sr'],
         neighbor_elements=['O'],
         cutoff=4,
         neighbor_num=12,
         defect=False)
nl.write('A.dat')

nl.build(center_elements=['Zr'],
         neighbor_elements=['Pb','Sr'],
         cutoff=4.5,
         neighbor_num=8,
         defect=False)
nl.write('BA.dat')
