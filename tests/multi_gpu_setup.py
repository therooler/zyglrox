from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.gates import *
import numpy as np

qc = QuantumCircuit(10, [Hadamard(wires=[i, ]) for i in np.random.randint(0, 9, 100)], device='GPU', ngpus=1, tensorboard=True)
output = qc.execute()
qc.initialize()
print(qc._sess.run(output))