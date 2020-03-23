from zyglrox.core._config import TF_COMPLEX_DTYPE
from zyglrox.core.gates import RX, ZZ, Hadamard
from zyglrox.core.utils import von_neumann_entropy, renyi_entropy, partial_trace
from zyglrox.core.circuit import QuantumCircuit
import tensorflow as tf
import numpy as np
import string

p = 3
N = 4
gates = []
gates.extend([Hadamard(wires=[i, ]) for i in range(N)])
for l in range(p):
    for j in range(0, N - 1, 2):
        gates.append(ZZ(wires=[j, j + 1]))

    for j in range(1, N - 1, 2):
        gates.append(ZZ(wires=[j, j + 1]))

    gates.append(ZZ(wires=[N - 1, 0]))
    gates.extend([RX(wires=[i, ]) for i in range(N)])

qc = QuantumCircuit(N, gates, circuit_order='layer', get_phi_per_layer=True, tensorboard=True)
# print(qc.phi_per_layer)
phi = qc.execute()
all_ptr_rhos = partial_trace(qc.phi_per_layer, [0, 1], [-1] + [2 for _ in range(N)])
# vn_entropies = von_neumann_entropy(all_ptr_rhos)
# ry_entropies = renyi_entropy(all_ptr_rhos, alpha=0.99)
qc.initialize()
lam, _= qc._sess.run(tf.linalg.eigh(all_ptr_rhos))
print(np.sum(lam,axis=1))
#
# rhos, vns, rys = qc._sess.run([all_ptr_rhos, vn_entropies, ry_entropies])
# for p in rhos:
#     print(p)
#     print(np.trace(p))
#
# print(vns)
# print(rys)
# phis = qc._sess.run(qc.phi_per_layer)
# for p in phis:
#     print(np.linalg.norm(p))
