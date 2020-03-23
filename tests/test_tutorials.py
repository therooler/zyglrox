from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.gates import Hadamard, PauliZ, Phase, CNOT
from zyglrox.core.observables import Observable, ExpectationValue
import numpy as np
import pytest


def test_tutorial_1():
    gates = [Hadamard(wires=[0, ]), PauliZ(wires=[0, ])]
    qc = QuantumCircuit(nqubits=1, gates=gates, device="CPU")
    qc.initialize()
    phi = qc.circuit(qc.phi)

    qc_tf = qc._sess.run(phi)
    obs = [Observable("x", wires=[0, ])]
    expval_layer = ExpectationValue(obs)
    measurements = qc._sess.run(expval_layer(phi))


def test_tutorial_2():
    gates = [Hadamard(wires=[0, ]), Phase(wires=[0, ], value=[np.pi / 8]),
             Hadamard(wires=[1, ]), Phase(wires=[1, ], value=[np.pi / 8]), CNOT(wires=[0, 1])]

    qc = QuantumCircuit(nqubits=2, gates=gates, tensorboard=True, device="CPU")

    phi = qc.circuit(qc.phi)
    qc.circuit.summary()
    obs = [Observable("X", wires=[0, ]), Observable("x", wires=[1, ]),
           Observable("y", wires=[0, ]), Observable("y", wires=[1, ]),
           Observable("z", wires=[0, ]), Observable("z", wires=[1, ])]
    expval_layer = ExpectationValue(obs)
    qc.initialize()
    measurements = qc._sess.run(expval_layer(phi))
