import numpy as np
from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.gates import RX, RY, CNOT
from zyglrox.models.qve import QuantumVariationalEigensolver, QuantumVariationalEigensolverGradFree
from zyglrox.core.hamiltonians import Hamiltonian, RandomFullyConnectedXYZ
from zyglrox.core.circuit_templates import circuit6
import matplotlib.pyplot as plt
import pytest

def test_bare_qve_example():
    np.random.seed(123213)
    N = 2
    epochs = 1000
    tol = 1e-6

    top = {0: [[0, 1]], 1: [[1, 0]]}
    interactions = {'x': {0: [[0]]}, 'y': {1: [[1]]}, 'z': {0: [[0]]}}
    params = {'x': -0.2, 'y': 0.5, 'z': 0.8}
    ham = Hamiltonian(top, interactions, params)

    gates = [RY(wires=[0, ]),
             RX(wires=[1, ]),
             CNOT(wires=[0, 1])]
    qc = QuantumCircuit(N, gates=gates, tensorboard=True)
    qve = QuantumVariationalEigensolver(ham, qc, exact=True)
    qc.circuit.summary()
    qve.train(max_iter=epochs, tol=tol)

    qve.get_convergence()
    qve.get_statistics()
    qve.get_wavefunction()
    plt.close()

def test_bare_qve_example_grad_free():

    N = 3

    hamiltonian = RandomFullyConnectedXYZ(L=N, seed=1337)

    gates= circuit6(N)

    qc = QuantumCircuit(N, gates=gates, tensorboard=False)
    qve = QuantumVariationalEigensolverGradFree(hamiltonian, qc, exact=True, device="CPU")
    qve.train(10)

    qve.get_convergence()
    qve.get_statistics()
    qve.get_wavefunction()
    plt.close()

