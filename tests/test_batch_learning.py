from zyglrox.core.observables import Observable, ExpectationValue
from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.gates import *
import pytest
import numpy as np


def test_batching_wave_functions():
    N = 3
    d = 13
    qc = QuantumCircuit(N, [Hadamard(wires=[0, ]),
                            Hadamard(wires=[1, ])], device="CPU")
    psi = tf.placeholder(dtype=tf.complex64, shape=(None, *[2 for _ in range(N)]))
    phi_dim_1 = qc.circuit(qc.phi)
    phi_batch = qc.circuit(psi)
    qc.initialize()
    phi_feed = np.zeros([2 for _ in range(N)])
    phi_feed[0,] = 1
    phi_feed = np.stack([phi_feed for _ in range(d)]).astype(np.complex64)
    out_1, out_batch = qc._sess.run([phi_dim_1, phi_batch], feed_dict={psi: phi_feed})


def test_batching_wave_functions_and_parameters():
    N = 3
    d = 13
    qc = QuantumCircuit(N, [RX(wires=[0, ]),
                            RX(wires=[1, ])], batch_params=True, device="CPU")
    psi = tf.placeholder(dtype=tf.complex64, shape=(None, *[2 for _ in range(N)]))
    theta = tf.convert_to_tensor([np.pi, np.pi])
    with pytest.raises(AssertionError, match="expected tensor with 3 dimensions, received tensor with shape"):
        qc.set_parameters(theta)
    qc.set_parameters(tf.reshape(theta, (1, 2, 1)))
    with pytest.raises(AssertionError,
                       match="We cannot have a batch of wave functions and a batch of parameters at the same time"):
        qc.circuit(psi)


def test_external_input_dtype():
    N = 3
    qc = QuantumCircuit(N, [RX(wires=[0, ]),
                            RX(wires=[1, ])], batch_params=True, device="CPU")
    with pytest.raises(TypeError, match="External input Tensor must have type "):
        theta = tf.placeholder(dtype=tf.complex64, shape=(None, 2, 1))
        qc.set_parameters(theta)
    with pytest.raises(TypeError, match="External input Tensor must have type "):
        theta = tf.placeholder(dtype=tf.float64, shape=(None, 2, 1))
        qc.set_parameters(theta)


def test_batching_learning_observables():
    N = 3
    d = 13
    qc = QuantumCircuit(N, [Hadamard(wires=[0, ]),
                            Hadamard(wires=[1, ])], device="CPU")
    psi = tf.placeholder(dtype=tf.complex64, shape=(None, *[2 for _ in range(N)]))
    phi_dim_1 = qc.circuit(qc.phi)
    phi_batch = qc.circuit(psi)
    qc.initialize()
    obs = [Observable("x", wires=[0, ]), Observable("x", wires=[1, ]), Observable("x", wires=[2, ]),
         Observable("y", wires=[0, ]), Observable("y", wires=[1, ]), Observable("y", wires=[2, ])]
    expval_layer = ExpectationValue(obs)
    phi_feed = np.zeros([2 for _ in range(N)])
    phi_feed[0,] = 1
    phi_feed = np.stack([phi_feed for _ in range(d)]).astype(np.complex64)
    out_1, out_batch = qc._sess.run([expval_layer(phi_dim_1), expval_layer(phi_batch)], feed_dict={psi: phi_feed})


def test_batching_works_correctly():
    N = 2
    qc = QuantumCircuit(N, [Hadamard(wires=[0, ]),
                            Hadamard(wires=[1, ])], device="CPU")
    obs = [Observable("x", wires=[0, ]), Observable("x", wires=[1, ])]
    phi = tf.placeholder(dtype=tf.complex64, shape=(None, *[2 for _ in range(N)]))
    expval_layer = ExpectationValue(obs)

    phi_batch = qc.circuit(phi)
    expvals = expval_layer(phi_batch)

    # four basis states
    states = np.eye(int(2 ** N))
    qc.initialize()
    out_batch = qc._sess.run([expvals], feed_dict={phi: states.reshape(-1, *[2 for _ in range(N)])})[0]
    out_batch = out_batch.reshape((-1,))
    assert np.allclose(out_batch, np.array([1, 1, 1, -1, -1, 1, -1, -1]))


def test_multiple_thetas():
    N = 3
    d = 13
    theta = tf.placeholder(dtype=tf.float32, shape=(None, 2, 1))
    qc = QuantumCircuit(N, [RX(wires=[0, ]),
                            RX(wires=[1, ])], batch_params=True, device="CPU")
    qc.set_parameters(theta)
    phi_dim_1 = qc.circuit(qc.phi)
    obs = [Observable("x", wires=[0, ]), Observable("x", wires=[1, ]), Observable("x", wires=[2, ]),
         Observable("y", wires=[0, ]), Observable("y", wires=[1, ]), Observable("y", wires=[2, ])]
    expval_layer = ExpectationValue(obs)

    expvals_dim_1 = expval_layer(phi_dim_1)
    qc.initialize()
    qc._sess.run([expvals_dim_1], feed_dict={theta: np.random.randn(d, 2, 1)})


def test_batch_params_argument():
    N = 3
    theta = tf.placeholder(dtype=tf.float32, shape=(None, 2, 1))
    qc = QuantumCircuit(N, [RX(wires=[0, ]),
                            RX(wires=[1, ])])
    with pytest.raises(AssertionError,
                       match=" pass the batch_params=True argument to the QuantumCircuit class to enable batches of parameters."):
        qc.set_parameters(theta)
        qc.circuit(qc.phi)


def test_batch_params_batch_dim():
    N = 3
    qc = QuantumCircuit(N, [RX(wires=[0, ]),
                            RX(wires=[1, ])], batch_params=True)
    with pytest.raises(AssertionError, match="batch_params=True, but no external input provided."):
        qc.circuit(qc.phi)
