import numpy as np

prefix = 'PTO'
nl_ao = np.loadtxt(f'{prefix}_AO-nl.dat')
nl_bo = np.loadtxt(f'{prefix}_BO-nl.dat')
nl_ba = np.loadtxt(f'{prefix}_BA-nl.dat')

np.savez(f'PTO-e_nl.npz', ao=nl_ao, bo=nl_bo, ba=nl_ba)