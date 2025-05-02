from ase.constraints import FixConstraint
import numpy as np
from ase.geometry import find_mic

class FixBondVectors(FixConstraint):
    maxiter: int = 500

    def __init__(self, pairs: list[int],
                 tolerance: float = 1e-8):
        self.pairs = pairs
        self.tolerance = tolerance
        self.initial_vectors = None

    def initialize_vectors(self, atoms):
        self.initial_vectors = []
        for pair in self.pairs:
            vector = atoms.get_distance(pair[0], pair[1], mic=True)
            self.initial_vectors.append(vector)
        self.initial_vectors = np.array(self.initial_vectors)
    
    def adjust_positions(self, atoms, new):
        
        if self.initial_vectors is None:
            self.initialize_vectors(atoms)
        
        print(f"Ref Bond Vectors: {self.initial_vectors}")
        print(f"New Bond Vectors: {new}")
        print(f"Adjusting..")
        masses = atoms.get_masses()
        cell = atoms.cell
        pbc = atoms.pbc

        for _ in range(self.maxiter):
            max_error = 0.0
            for i, (a, b) in enumerate(self.pairs):
                # 计算当前键矢量（考虑周期性边界条件）
                d = new[b] - new[a]
                d_mic, _ = find_mic(d, cell, pbc)
                
                # 目标方向为初始单位向量，长度保持为当前键长
                target_vector = self.initial_vectors[i]
                
                # 计算需要调整的总矢量差
                delta_total = target_vector - d_mic
                
                # 按质量分配调整量
                m_a, m_b = masses[a], masses[b]
                total_mass = m_a + m_b
                delta_a = (m_b / total_mass) * delta_total
                delta_b = -(m_a / total_mass) * delta_total
                
                # 更新位置
                new[a] += delta_a
                new[b] += delta_b
                
                # 计算方向误差（1 - cosθ）
                adjusted_d = new[b] - new[a]
                adjusted_d_mic, _ = find_mic(adjusted_d, cell, pbc)
                adjusted_direction = adjusted_d_mic / np.linalg.norm(adjusted_d_mic)
                cos = np.dot(adjusted_direction, self.initial_vectors[i]  / np.linalg.norm(self.initial_vectors[i]))
                error = 1 - cos
                max_error = max(max_error, error)

            if max_error < self.tolerance:
                print(f"Converged after {_} iterations")
                print(f"Adjusted Bond Vectors: {new}")
                break
        
        else:
            raise RuntimeError("FixBondVectors: not converged after {} iterations".format(self.maxiter))

    def adjust_forces(self, atoms, forces):
        raise NotImplementedError("FixBondVectors: adjust_forces not implemented")