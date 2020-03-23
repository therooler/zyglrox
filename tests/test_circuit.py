from zyglrox.core.observables import Observable, ExpectationValue
from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.gates import *
import pennylane as qml
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def test_hadamard():
    dev = qml.device("default.qubit", analytic=True, wires=1)

    @qml.qnode(dev)
    def circuitx():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliX(0))

    @qml.qnode(dev)
    def circuity():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliY(0))

    @qml.qnode(dev)
    def circuitz():
        qml.Hadamard(wires=0)
        return qml.expval(qml.PauliZ(0))

    qc_pl = np.array([circuitx(), circuity(), circuitz()])
    qc = QuantumCircuit(1, [Hadamard(wires=[0, ])], device='CPU')
    obs = [Observable("x", wires=[0, ]), Observable("y", wires=[0, ]), Observable("z", wires=[0, ])]
    phi = qc.circuit(qc.phi)
    expval_layer = ExpectationValue(obs)
    qc.initialize()
    qc_tf = qc._sess.run(expval_layer(phi))
    qc_tf = qc_tf.flatten()
    assert all(np.isclose(qc_pl, qc_tf))


def test_pauli_x():
    dev = qml.device("default.qubit", analytic=True, wires=1)

    @qml.qnode(dev)
    def circuitx():
        qml.PauliX(wires=0)
        return qml.expval(qml.PauliX(0))

    @qml.qnode(dev)
    def circuity():
        qml.PauliX(wires=0)
        return qml.expval(qml.PauliY(0))

    @qml.qnode(dev)
    def circuitz():
        qml.PauliX(wires=0)
        return qml.expval(qml.PauliZ(0))

    qc_pl = np.array([circuitx(), circuity(), circuitz()])
    qc = QuantumCircuit(1, [PauliX(wires=[0, ])], device="CPU")
    obs = [Observable("x", wires=[0, ]), Observable("y", wires=[0, ]), Observable("z", wires=[0, ])]
    phi = qc.circuit(qc.phi)
    expval_layer = ExpectationValue(obs)
    qc.initialize()
    qc_tf = qc._sess.run(expval_layer(phi))
    qc_tf = qc_tf.flatten()
    assert all(np.isclose(qc_pl, qc_tf))


def test_rot_x():
    dev = qml.device("default.qubit", analytic=True, wires=1)
    theta = 0.3 * np.math.pi

    @qml.qnode(dev)
    def circuitx():
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliX(0))

    @qml.qnode(dev)
    def circuity():
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliY(0))

    @qml.qnode(dev)
    def circuitz():
        qml.RX(theta, wires=0)
        return qml.expval(qml.PauliZ(0))

    qc_pl = np.array([circuitx(), circuity(), circuitz()])
    qc = QuantumCircuit(1, [RX(wires=[0, ], value=[theta])],device="CPU")
    obs = [Observable("x", wires=[0, ]), Observable("y", wires=[0, ]), Observable("z", wires=[0, ])]
    phi = qc.circuit(qc.phi)
    expval_layer = ExpectationValue(obs)
    qc.initialize()
    qc_tf = qc._sess.run(expval_layer(phi))
    qc_tf = qc_tf.flatten()
    assert all(np.isclose(qc_pl, qc_tf))


def test_cnot():
    dev = qml.device("default.qubit", analytic=True, wires=2)

    @qml.qnode(dev)
    def circuitx():
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliX(0))

    @qml.qnode(dev)
    def circuity():
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliY(0))

    @qml.qnode(dev)
    def circuitz():
        qml.PauliX(wires=0)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    qc_pl = np.array([circuitx(), circuity(), circuitz()])
    qc = QuantumCircuit(2, [PauliX(wires=[0, ]), CNOT(wires=[0, 1])],device="CPU")
    obs = [Observable("x", wires=[0, ]), Observable("y", wires=[0, ]), Observable("z", wires=[0, ])]
    phi = qc.circuit(qc.phi)
    expval_layer = ExpectationValue(obs)
    qc.initialize()
    qc_tf = qc._sess.run(expval_layer(phi))
    qc_tf = qc_tf.flatten()
    assert all(np.isclose(qc_pl, qc_tf))


def test_toffoli():
    dev = qml.device("default.qubit", analytic=True, wires=3)

    @qml.qnode(dev)
    def circuitx():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.Toffoli(wires=[0,1,2])
        return qml.expval(qml.PauliX(2))

    @qml.qnode(dev)
    def circuity():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.Toffoli(wires=[0, 1, 2])
        return qml.expval(qml.PauliY(2))

    @qml.qnode(dev)
    def circuitz():
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.Hadamard(wires=2)
        qml.Toffoli(wires=[0, 1, 2])
        return qml.expval(qml.PauliZ(2))

    qc_pl = np.array([circuitx(), circuity(), circuitz()])
    qc = QuantumCircuit(3, [Hadamard(wires=[0,]),Hadamard(wires=[1,]),Hadamard(wires=[2,]), Toffoli(wires=[0, 2,1 ])],device="CPU")
    obs = [Observable("x", wires=[2, ]), Observable("y", wires=[2, ]), Observable("z", wires=[2, ])]
    phi = qc.circuit(qc.phi)
    expval_layer = ExpectationValue(obs)
    qc.initialize()
    qc_tf = qc._sess.run(expval_layer(phi))
    qc_tf = qc_tf.flatten()
    assert all(np.isclose(qc_pl, qc_tf))

def test_rotations_and_cnot():
    dev = qml.device("default.qubit", analytic=True, wires=2)
    theta = 0.1
    phi = 0.2

    @qml.qnode(dev)
    def circuitx():
        qml.RX(theta, wires=0)
        qml.RY(phi, wires=1)
        qml.CNOT(wires=[1, 0])
        return qml.expval(qml.PauliX(0))

    @qml.qnode(dev)
    def circuity():
        qml.RX(theta, wires=0)
        qml.RY(phi, wires=1)
        qml.CNOT(wires=[1, 0])
        return qml.expval(qml.PauliY(0))

    @qml.qnode(dev)
    def circuitz():
        qml.RX(theta, wires=0)
        qml.RY(phi, wires=1)
        qml.CNOT(wires=[1, 0])
        return qml.expval(qml.PauliZ(0))

    qc_pl = np.array([circuitx(), circuity(), circuitz()])
    qc = QuantumCircuit(2, [RX(wires=[0, ], value=[theta]),
                            RY(wires=[1, ], value=[phi]),
                            CNOT(wires=[1, 0])],device="CPU")
    obs = [Observable("x", wires=[0, ]), Observable("y", wires=[0, ]), Observable("z", wires=[0, ])]
    phi = qc.circuit(qc.phi)
    expval_layer = ExpectationValue(obs)
    qc.initialize()
    qc_tf = qc._sess.run(expval_layer(phi))
    qc_tf = qc_tf.flatten()
    assert all(np.isclose(qc_pl, qc_tf))


def test_r3():
    dev = qml.device("default.qubit", analytic=True, wires=2)
    theta = 0.2
    phi = 0.2
    delta = 0.2

    @qml.qnode(dev)
    def circuitx():
        qml.Rot(theta, phi, delta, wires=0)
        return qml.expval(qml.PauliX(0))

    @qml.qnode(dev)
    def circuity():
        qml.Rot(theta, phi, delta, wires=0)
        return qml.expval(qml.PauliY(0))

    @qml.qnode(dev)
    def circuitz():
        qml.Rot(theta, phi, delta, wires=0)
        return qml.expval(qml.PauliZ(0))

    qc_pl = np.array([circuitx(), circuity(), circuitz()])
    qc = QuantumCircuit(2, [R3(wires=[0, ], value=[theta, phi, delta])],device="CPU")
    obs = [Observable("x", wires=[0, ]), Observable("y", wires=[0, ]), Observable("z", wires=[0, ])]
    phi = qc.circuit(qc.phi)
    expval_layer = ExpectationValue(obs)
    qc.initialize()
    qc_tf = qc._sess.run(expval_layer(phi))
    qc_tf = qc_tf.flatten()
    assert all(np.isclose(qc_pl, qc_tf))


def test_swap():
    dev = qml.device("default.qubit", analytic=True, wires=2)
    theta = 0.2
    phi = 0.2
    delta = 0.2

    @qml.qnode(dev)
    def circuit():
        qml.Rot(theta, phi, delta, wires=0)
        qml.Rot(theta, phi, delta, wires=1)
        qml.SWAP(wires=[0, 1])
        return (qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1)),)

    qc = QuantumCircuit(2, [R3(wires=[0, ], value=[theta, phi, delta]),
                            R3(wires=[1, ], value=[theta, phi, delta]),
                            Swap(wires=[0, 1])], device="CPU")
    obs = [Observable("x", wires=[0, ]), Observable("x", wires=[1, ])]
    psi = qc.circuit(qc.phi)
    expval_layer = ExpectationValue(obs)
    qc.initialize()
    qc_tf = qc._sess.run(expval_layer(psi))
    qc_tf = qc_tf.flatten()
    qc_pl = circuit()
    assert all(np.isclose(qc_pl, qc_tf))


def test_phase():
    dev = qml.device("default.qubit", analytic=True, wires=1)
    theta = 0.2

    @qml.qnode(dev)
    def circuit():
        qml.PhaseShift(theta, wires=[0, ])
        return (qml.expval(qml.PauliX(0)))

    qc = QuantumCircuit(2, [Phase(wires=[0, ], value=[theta, ]),],device="CPU")
    obs = [Observable("x", wires=[0, ])]
    phi = qc.circuit(qc.phi)
    expval_layer = ExpectationValue(obs)
    qc.initialize()
    qc_tf = qc._sess.run(expval_layer(phi))
    qc_tf = qc_tf.flatten()
    qc_pl = circuit()
    assert all(np.isclose(qc_pl, qc_tf))


def test_same_wire_gates():
    qc = QuantumCircuit(2, [RX(wires=[0, ]) for i in range(10)],device="CPU")
    qc.initialize()
    qc.circuit(qc.phi)
