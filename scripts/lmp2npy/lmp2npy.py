import numpy as np 

def collect_energy(file_name: str):
    data = np.loadtxt(file_name, comments="#", usecols=(-1))
    nframe = data.shape[0]
    energy = data.reshape((nframe, 1))
    print(f"Number of frames: {nframe}")
    print(f"Energy shape: {energy.shape}")
    return energy

def collect_forces(dump_file: str):

    def get_natoms():
        with open(dump_file, 'r') as f:
            for line in f:
                if line.startswith('ITEM: NUMBER OF ATOMS'):
                    return int(f.readline().strip())

    def get_forces(dump_file: str, natom: int):
        forces = []
        with open(dump_file, 'r') as f:
            for line in f:
                if line.startswith('ITEM: ATOMS id type x y z fx fy fz'):
                    tmp = []
                    for _ in range(natom):
                        tmp.append([float(x) for x in f.readline().strip().split()[3:6]])
                    forces.append(tmp)
        return np.array(forces)
    natom = get_natoms()
    forces = get_forces(dump_file, natom)
    print(f"Force shape: {forces.shape}")

    return forces

def main():
    e = collect_energy('pe.dat')
    f = collect_forces('pe.lammpstrj')
    np.save('energy.npy', e)
    np.save('forces.npy', f)

if __name__ == "__main__":
    main()