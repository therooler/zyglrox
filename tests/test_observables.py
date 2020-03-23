from zyglrox.core.observables import Observable, observable_dict
import pytest
import numpy as np
import tensorflow as tf
def test_standard_observables():
    for name in observable_dict.keys():
        Observable(name, wires=[0,])

def test_wrong_observable_name():
    with pytest.raises(AssertionError, match='not recognized, choose from'):
        Observable("PauliX", wires=[0,])

def test_gate_class_correct_phi():
    wrong_phi = tf.convert_to_tensor(np.ones(20), dtype=tf.complex64)
    for name in observable_dict.keys():
        obs = Observable(name, wires=[0, ])
        with pytest.raises(AssertionError, match='Input shape must have at least'):
            obs(wrong_phi)

def test_gate_class_correct_phi_2():
    wrong_phi = tf.convert_to_tensor(np.ones((1,20)), dtype=tf.complex64)
    for name in observable_dict.keys():
        obs = Observable(name, wires=[0, ])
        with pytest.raises(AssertionError, match='Input vector needs to have shape'):
            obs(wrong_phi)