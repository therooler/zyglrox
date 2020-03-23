import tensorflow as tf
import numpy as np
from typing import List
from contextlib import redirect_stdout
import os

from zyglrox.core.observables import ExpectationValue
from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.hamiltonians import Hamiltonian
from zyglrox.core._config import TF_FLOAT_DTYPE

class ModelAgnosticMetaLearner(object):
    """"""

    def __init__(self, hamiltonian: List[Hamiltonian], qc: List[QuantumCircuit], energy_fn, parameter_fn,
                 exact=False, device="CPU", optimizer=None, **params):
        """"""
        assert ((len(qc) == 2) & (isinstance(qc, List))), "qc must be a list of length 2, found {}".format(qc)
        assert all(isinstance(q, QuantumCircuit) for q in
                   qc), "type of all objects in qc must be 'QuantumCircuit', received {}".format([type(q) for q in qc])

        assert ((len(hamiltonian) == 2) & (
            isinstance(hamiltonian, List))), "hamiltonian must be a list of length 2, found {}".format(hamiltonian)

        self.hamiltonian_min = hamiltonian[0]
        self.hamiltonian_max = hamiltonian[1]
        self.qc_a = qc[0]
        self.qc_b = qc[1]
        assert self.qc_a.nqubits == self.qc_b.nqubits, "the supplied circuits are defined on a different number of qubits, found {} and {}".format(
            self.qc_a.nqubits, self.qc_b.nqubits)
        assert self.qc_a.nparams == self.qc_b.nparams, "the supplied circuits have a different number of parameters, found {} and {}".format(
            self.qc_a.nparams, self.qc_b.nparams)
        # get all the non-zero fields from the hamiltonian
        self.obs = self.hamiltonian_min.get_observables()
        # set the hamiltonian terms

        self.exact = exact
        self.TRAIN = False
        self.ALPHA = params.get("ALPHA", 0.05)
        self.BETA = params.get("BETA", 0.05)
        if optimizer is None:
            optimizer = tf.compat.v1.train.AdamOptimizer(self.BETA)
        else:
            assert isinstance(optimizer,
                              tf.compat.v1.optimizers.Optimizer), "Passed optimizer must be a tensorflow 'Optimizer' object, received {}".format(
                type(optimizer))
        self.NUMSAMPLES = params.get("num_samples", 5)
        self.PARIT = params.get("PARIT", 2)
        self.model_name = params.get("model_name", 'MAML')
        self.save_model = params.get("save_model", False)
        self.load_model = params.get("load_model", False)
        self.tfcheckpoint_path = params.get("tfcheckpoint_path", "./tfcheckpoints")

        self.energy_fn = energy_fn
        self.parameter_fn = parameter_fn

        self._build_graph(optimizer, device)

    def _build_graph(self, optimizer, device="CPU"):
        """
        Build the computational tensorflow graph.

        Args:
            *optimizer (tf.optimizers.Optimizer)*:
                Desired optimizer to perform the QVE algorithm.

            *device (string)*:
                Device of choice for running tensorflow.

        Returns (inplace):
            None

        """
        assert device in ['CPU', 'GPU'], "device must be one of '['CPU', 'GPU'], received {}".format(device)

        if self.exact:
            assert self.qc_a.nqubits < 18, "Cannot perform exact diagonalization for {} qubits, max is 18".format(
                self.qc_a.nqubits)
            with tf.name_scope("exact_hamiltonian"):
                self.hamiltonian_min.get_hamiltonian()
                self.hamiltonian_max.get_hamiltonian()
                self.e_min = self.hamiltonian_min.energies[0]
                self.e_max = self.hamiltonian_max.energies[0]
        self.theta_var = tf.Variable(tf.random.uniform((1, self.qc_a.nparams, 1), seed=10) * 2 * np.pi)

        def task(parameters):
            with tf.name_scope("circuit_A"):
                theta_input = tf.identity(self.theta_var)
                self.qc_a.set_parameters(self.theta_var)
                phi_a = self.qc_a.circuit(self.qc_a.phi)

                expval_layer_a = ExpectationValue(self.obs)
                expvals_a = expval_layer_a(phi_a)

                energy_a = self.energy_fn(expvals_a, parameters)

                grads_a = tf.stop_gradient(tf.gradients(energy_a, self.theta_var))
            with tf.name_scope("circuit_B"):
                theta_prime = theta_input - self.ALPHA * grads_a[0]
                self.qc_b.set_parameters(theta_prime)
                phi_b = self.qc_b.circuit(self.qc_b.phi)

                expval_layer_b = ExpectationValue(self.obs)
                expvals_b = expval_layer_b(phi_b)

                energy_b = tf.reduce_mean(self.energy_fn(expvals_b, parameters))

            return energy_a, energy_b

        tau = self.parameter_fn()
        self.en_a, self.en_b = tf.map_fn(task, tau, parallel_iterations=self.PARIT, dtype=(TF_FLOAT_DTYPE, TF_FLOAT_DTYPE))
        self.train_step = optimizer.minimize(self.en_b, var_list=self.theta_var)

        if self.save_model:
            self.save_path = self.tfcheckpoint_path + "/" + self.model_name
            if not os.path.exists(self.save_path):
                os.mkdir(self.save_path)
            self.saver = self.load_path = tf.compat.v1.train.Saver([self.theta_var], max_to_keep=4)
        self.qc_a.initialize()
        if self.load_model:
            assert os.path.exists(self.load_path), "path {} does not exist, save a model here first".format(
                self.load_path)
            out = self.saver.restore(self.qc_a._sess, self.load_path)
            print("Loaded model from {}".format(out))

    def train(self, epochs=1000, tol=1e-8):
        r"""
        Train the quantum variational eigensolver. We minimize the energy :math:`\langle H \rangle_\theta` as defined above.

        Args:
            *epochs (int)*:
                Number of max iterations for the training algorithm.

            *tol (float)*:
                Tolerance on the energy. If the absolute difference between iterations is smaller than this value, training stops.

        Returns expvals (np.ndarray):
            Expectation values after training is complete.

        """
        self.TRAIN = True
        self.CONVERGED = False
        self.energy_a_per_epoch = []
        self.energy_b_per_epoch = []
        E_0 = None
        for ep in range(epochs):
            _, energy_a, energy_b, = self.qc_a._sess.run([self.train_step, self.en_a, self.en_b])
            energy_a = np.mean(energy_a)
            energy_b = np.mean(energy_b)
            self.energy_a_per_epoch.append(energy_a)
            self.energy_b_per_epoch.append(energy_b)
            if ep == 0:
                E_0 = energy_b
            else:
                delta_E = np.abs(E_0 - energy_b)
                E_0 = energy_b

                if not (ep % 50):
                    if self.save_model:
                        out = self.saver.save(self.qc_a._sess, self.save_path + "/cptk")
                        print("Model saved in {}".format(out))
                        params = self.qc_a._sess.run(self.theta_var)
                        np.save(self.save_path + "/model_parameters", params)

                    print("\n\tQVE Epoch {}: \t Energy : {}".format(ep, energy_b))
                if delta_E < tol:
                    print("Converged after {} epochs \n".format(ep))
                    self.CONVERGED = True
                    break
        if self.save_model:
            if self.exact:
                exact_energy_min, exact_energy_max = self.qc_a._sess.run([self.e_min, self.e_max])
                np.save(self.save_path + "/e_min", exact_energy_min)
                np.save(self.save_path + "/e_max", exact_energy_max)
            np.save(self.save_path + "/energy_a", np.array(self.energy_a_per_epoch))
            np.save(self.save_path + "/energy_b", np.array(self.energy_b_per_epoch))
            with open(self.save_path + "/model_specifications.txt", 'w') as f:
                f.write("Model name = {}\n".format(self.model_name))
                f.write("Number of qubits = {}\n".format(self.qc_a.nqubits))
                f.write("Number of model parameters = {}\n".format(self.qc_a.nparams))
                f.write("Number of samples used = {}\n".format(self.NUMSAMPLES))
                f.write("Alpha learning rate = {}\n".format(self.ALPHA))
                f.write("Beta learning rate = {}\n".format(self.BETA))
                f.write("Maximum number of epochs = {}\n".format(epochs))
                f.write("Convergence tolerance = {}\n".format(tol))
                f.write("Number of parallel iterations = {}\n".format(self.PARIT))
                f.write("Model converged = {}\n".format(self.CONVERGED))
                with redirect_stdout(f):
                    self.qc_a.circuit.summary()
