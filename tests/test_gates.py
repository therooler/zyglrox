from zyglrox.core.gates import Hadamard, CNOT, PauliX, PauliY, PauliZ, RX, RY, RZ, \
    R3, CRX, CRY, CRZ, CR3, CZ, Swap, Toffoli
import pytest
import numpy as np
import tensorflow as tf


@pytest.fixture
def all_gates_list():
    return [Hadamard, CNOT, PauliX, PauliY, PauliZ, RX, RY, RZ, R3, CRX, CRY, CRZ, CR3, CZ, Swap]


@pytest.fixture
def fixed_gates_list_single():
    return [Hadamard, PauliX, PauliY, PauliZ]


@pytest.fixture
def fixed_gates_list_double():
    return [CNOT, Swap, CZ]


@pytest.fixture
def parametrized_gates_list_single():
    return [RX, RY, RZ, R3]


@pytest.fixture
def parametrized_gates_list_double():
    return [CRX, CRY, CRZ, CR3]


def test_wires(fixed_gates_list_single, fixed_gates_list_double, parametrized_gates_list_single,
               parametrized_gates_list_double):
    for gate in fixed_gates_list_single + parametrized_gates_list_single:
        with pytest.raises(AssertionError):
            gate(wires=[0, 1])
        with pytest.raises(AssertionError):
            gate(wires=[0, 1, 2])

    for gate in fixed_gates_list_double + parametrized_gates_list_double:
        with pytest.raises(AssertionError):
            gate(wires=[0, ])
        with pytest.raises(AssertionError):
            gate(wires=[0, 1, 2])


def test_nparams(fixed_gates_list_single, fixed_gates_list_double, parametrized_gates_list_single,
                 parametrized_gates_list_double):
    for gate in fixed_gates_list_single:
        gate = gate(wires=[0, ])
        assert gate.nparams == 0
    for gate in fixed_gates_list_double:
        gate = gate(wires=[0, 1])
        assert gate.nparams == 0
    for gate in parametrized_gates_list_single:
        gate = gate(wires=[0, ])
        assert gate.nparams > 0
    for gate in parametrized_gates_list_double:
        gate = gate(wires=[0, 1])
        assert gate.nparams > 0


def test_gate_class_wires_list(all_gates_list):
    for gate in all_gates_list:
        with pytest.raises(AssertionError, match="'wires' must be a list"):
            gate(wires=np.array([0, 2]))


def test_gate_class_correct_phi(fixed_gates_list_single, fixed_gates_list_double, parametrized_gates_list_single,
                                parametrized_gates_list_double):
    wrong_phi = tf.convert_to_tensor(np.ones((1,20)), dtype=tf.complex64)
    for gate in fixed_gates_list_single + parametrized_gates_list_single:
        g = gate(wires=[0, ])
        with pytest.raises(AssertionError, match="Input vector needs to have shape"):
            g(wrong_phi)
    for gate in fixed_gates_list_double + parametrized_gates_list_double:
        g = gate(wires=[0, 1])
        with pytest.raises(AssertionError, match="Input vector needs to have shape"):
            g(wrong_phi)


def test_gate_class_kwarges(fixed_gates_list_single, fixed_gates_list_double, parametrized_gates_list_single,
                            parametrized_gates_list_double):
    for gate in fixed_gates_list_single + parametrized_gates_list_single:
        with pytest.raises(AssertionError, match="is not an allowed keyword argument"):
            g = gate(wires=[0, ], trainable=False, dummy_kwargs = None)
    for gate in fixed_gates_list_double + parametrized_gates_list_double:
        with pytest.raises(AssertionError, match="is not an allowed keyword argument"):
            g = gate(wires=[0, 1], trainable=False, dummy_kwargs = None)
