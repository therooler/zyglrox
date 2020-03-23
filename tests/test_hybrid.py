from zyglrox.core.observables import Observable, ExpectationValue
from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.gates import *
import pytest
import numpy as np


def test_hybrid_neural_network_gradient_tracking():
    N = 2
    gates = [RX(wires=[0, ]),
             RX(wires=[1, ])]
    qc = QuantumCircuit(N, gates=gates, tensorboard=True, device="CPU")

    # Make a neural network
    NN = tf.keras.Sequential([tf.keras.layers.Dense(qc.nparams * 10, activation=tf.keras.activations.tanh),
                              tf.keras.layers.Dense(2, activation=tf.keras.activations.tanh),
                              tf.keras.layers.Lambda(lambda x: x * 2 * np.pi)
                              ])
    x_in = tf.ones((qc.nparams, 1)) * 0.01
    theta = NN(x_in)
    theta = tf.reduce_mean(theta, axis=0)
    theta = tf.reshape(theta, (1, -1, 1))

    qc.set_parameters(theta)
    phi = qc.circuit(qc.phi)
    obs = Observable("x", wires=[0, ]), Observable("x", wires=[1, ]), Observable("z", wires=[1, ])
    expval_layer = ExpectationValue(obs)
    expvals = expval_layer(phi)

    # Get energy for each timstep
    energy = tf.reduce_sum(expvals)

    opt = tf.compat.v1.train.AdamOptimizer(0.0001)
    grads = opt.compute_gradients(energy)
    qc.initialize()

    g = qc._sess.run(grads)
