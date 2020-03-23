# Copyright 2020 Roeland Wiersema
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import itertools as it
from typing import Union

from mpl_toolkits.axes_grid1 import make_axes_locatable

from zyglrox.core.observables import ExpectationValue
from zyglrox.core.utils import flatten
from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.hamiltonians import Hamiltonian
from zyglrox.core._config import TF_FLOAT_DTYPE, TF_COMPLEX_DTYPE

class QuantumVariationalEigensolver(object):
    """
    Quantum Variational Eigensolver according to  `Peruzzo et al. (2014) <https://arxiv.org/abs/1304.3061>`_ with gradient based
    optimization.

    """

    def __init__(self, hamiltonian: Hamiltonian, qc: QuantumCircuit, exact=False, optimizer=None, **params):
        r"""
        We take a quantum circuit with parameters :math:`\theta` which
        implements some unitary transformation :math:`\mathcal{U}(\theta)`. This unitary brings our initial state
        :math:`|0\rangle` to a state :math:`|\psi(\theta)\rangle`.

        .. math::

            \mathcal{U}(\theta)|0\rangle = |\psi(\theta)\rangle

        Then, we realize that any Hamiltonian :math:`H` can be written as a polynomial of single qubit interactions.

        .. math::

            H = \sum_{i\alpha} h_\alpha^i \sigma_i^\alpha +  \sum_{ij\alpha\beta} w_{\alpha\beta}^{ij} \sigma_{i}^{\alpha}\sigma_{j}^{\beta} + \ldots

        where :math:`i,j,\ldots=1,\ldots,N` indicate the qubit and :math:`\alpha,\beta,\ldots=x,y,z` indicates the Pauli operator.
        By measuring the expectation values of the quantum circuit, we obtain a variational energy which is a lower bound on
        the true energy :math:`\langle H \rangle = \langle \psi| H |\psi \rangle`.

        .. math::

            \langle H \rangle \approx  \langle H \rangle_{\theta} = \sum_{i\alpha} h_\alpha^i \langle\sigma_i^\alpha\rangle +
            \sum_{ij\alpha\beta} w_{\alpha\beta}^{ij} \langle \sigma_{i}^{\alpha} \sigma_{j}^{\beta}\rangle + \ldots

        By minimizing this variational energy, we hope to learn a quantum circuit which best represents the groundstate of
        :math:`H`. If this minimization is successful, we can easily sample the ground state.

        For the Gradient-based QVE, we calculate the gradient for each parametrized unitary in the quantum circuit as follows:

        .. math::

            \frac{\partial \mathcal{U}(\theta) }{\partial \theta_j} = \left(\prod_{i=1}^{j-1} \mathcal{U}_i(\theta_i)\right) \frac{\partial \mathcal{U}_j(\theta_j) }{\partial \theta_j}
            \left(\prod_{i=j+1}^{M} \mathcal{U}_i(\theta_i)\right)

        Args:
            *hamiltonian (list)*:
                List of numpy arrays containing the operators and their locations. See :func:`~zyglrox.core.observables.verify_hamiltonian`
                for the required format.

            *qc (QuantumCircuit)*:
                ``QuantumCircuit`` object containing a quantum circuit architecture.

            *exact (bool)*:
                Boolean indicating whether we want to calculate the exact ground state and eigenvectors of the Hamiltonian.
                Will throw an error if the number of qubits is larger than 18.

            *optimizer (tf.optimizers.Optimizer)*:
                Desired optimizer to perform the QVE algorithm.

        Returns (inplace):
            None

        """
        assert isinstance(qc, QuantumCircuit), "qc must be a 'QuantumCircuit' object, received {}".format(type(qc))
        # assert verify_hamiltonian(hamiltonian)
        self.hamiltonian = hamiltonian
        # get all the non-zero fields from the hamiltonian
        self.obs = hamiltonian.get_observables()
        # set the hamiltonian terms
        self.hamiltonian_terms = hamiltonian.get_hamiltonian_terms()

        self.hamiltonian = hamiltonian
        self.qc = qc
        self.exact = exact
        self.TRAIN = False

        assert self.hamiltonian.nsites == self.qc.nqubits, "hamiltonian defined for {}, but quantum circuit has only {} qubits".format(
            self.hamiltonian.nsites, self.qc.nqubits)

        self.epsilon = params.get("epsilon", 0.05)
        self.model_name = params.get("model_name", 'QVE')
        self.save_model = params.get("save_model", False)
        self.load_model = params.get("load_model", False)
        self.verbose = params.get("verbose", True)
        self.feed_in_hamiltonian_terms = params.get("feed_in_hamiltonian_terms", False)
        self.tfcheckpoint_path = params.get("tfcheckpoint_path", "./tfcheckpoints")
        if optimizer is None:
            optimizer = tf.compat.v1.train.AdamOptimizer(self.epsilon)
        # else:
        #     assert isinstance(optimizer,
        #                       tf.keras.optimizers.Optimizer), "Passed optimizer must be a tensorflow 'Optimizer' object, received {}".format(
        #         type(optimizer))
        self._build_graph(optimizer)

    def _build_graph(self, optimizer):
        """
        Build the computational tensorflow graph.

        Args:
            *optimizer (tf.optimizers.Optimizer)*:
                Desired optimizer to perform the QVE algorithm.

        Returns (inplace):
            None

        """
        if self.exact:
            with tf.name_scope("exact_hamiltonian"):
                self.hamiltonian.get_hamiltonian()
                self.gs_energy = self.hamiltonian.energies[0]
                self.gs_wavefunction = self.hamiltonian.gs

        phi = self.qc.circuit(self.qc.phi)

        with tf.name_scope("train_step"):
            self.optimizer = optimizer
            self.expectation_layer = ExpectationValue(self.obs)
            self.expvals = self.expectation_layer(phi)
            if self.feed_in_hamiltonian_terms:
                self.hamiltonian_terms_feed = tf.compat.v1.placeholder(dtype=TF_FLOAT_DTYPE,
                                                                       shape=len(self.hamiltonian_terms),
                                                                       name='terms')
                self.energy = tf.reduce_sum(flatten(self.hamiltonian_terms_feed) * flatten(self.expvals), name="energy")
            else:
                self.energy = tf.reduce_sum(
                    tf.constant(self.hamiltonian_terms.flatten(), dtype=TF_FLOAT_DTYPE) * flatten(self.expvals),
                    name="energy")
            self.train_step = self.optimizer.minimize(self.energy)

        if self.save_model:
            self.saver = tf.compat.v1.train.Saver([self.qc.circuit.trainable_variables], max_to_keep=4)
        self.qc.initialize()
        if self.load_model:
            out = self.saver.restore(self.qc._sess, self.tfcheckpoint_path + self.model_name)
            print("Loaded model from {}".format(out))

    def train(self, epochs=1000, tol=1e-8, fetch_from_graph={}) -> np.ndarray:
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
        self.energy_per_epoch = []
        self.epochs_converged = epochs
        expvals = None
        energy_buffer = np.zeros(11)
        for ep in range(epochs):
            if self.feed_in_hamiltonian_terms:
                expvals, energy, _ = self.qc._sess.run([self.expvals, self.energy, self.train_step], feed_dict={
                    self.hamiltonian_terms_feed: self.hamiltonian_terms.flatten()})
            else:
                expvals, energy, _ = self.qc._sess.run([self.expvals, self.energy, self.train_step])
            self.energy_per_epoch.append(energy)

            energy_buffer[ep % 11] = energy
            if not (ep % 50):
                if self.save_model:
                    out = self.saver.save(self.qc._sess, self.tfcheckpoint_path)
                    if self.verbose:
                        print("Model saved in {}".format(out))
                print("\n\tQVE Epoch {}: \t Energy : {}".format(ep, energy))
            if not ep % 11:
                if np.abs(energy_buffer[0] - energy_buffer[10]) < tol:
                    if self.verbose:
                        print("Converged after {} epochs \n".format(ep))
                    self.CONVERGED = True
                    self.epochs_converged = ep
                    break

        return expvals

    def get_convergence(self, plot=True) -> Union[float, None]:
        r"""
        Plot the energy at each epoch. If the argument `exact=True` was passed to the``QuantumVariationalEigensolverGradFree`` constructor we also
        calculates the residual energy :math:`\epsilon_{\text{res}} = |\text{min}(E_{qc} - E_{gs_wavefunction})|`. Otherwise returns None

        Args:
            *plot (bool)*:
                Wheter to plot the training schedule.

        Returns (float):
            Residual energy.

        """
        assert self.TRAIN, "Train the model before plotting"
        res = None
        grid = list(range(len(self.energy_per_epoch)))
        if plot:
            plt.plot(grid, np.array(self.energy_per_epoch).flatten(), label='QVE')
            if self.exact:
                res = np.abs(np.min(self.energy_per_epoch) - self.gs_energy)
                print("Residual Energy = {}".format(res))
                plt.ylim([self.gs_energy - 1, 0])
                plt.plot(grid, [self.gs_energy for _ in range(len(self.energy_per_epoch))], label='Exact')
            plt.xlabel('iteration')
            plt.ylabel(r'$E=\langle \psi_\theta|H|\psi_\theta\rangle$')
            plt.title("Convergence of Quantum Variational Eigensolver")
            plt.legend()

        return res

    def get_statistics(self, plot=True):
        r"""
        Compare the ground state statistics of the ``QuantumCircuit`` with the exact ground state statistics.

        .. math::

            \text{MSE}_{h} = \frac{1}{M} \sum_i^M (\langle \sigma_i \rangle - \langle \hat{\sigma}_i \rangle)^2 \\
            \text{MSE}_{w} = \frac{1}{M} \sum_{i,j}^M (\langle \sigma_i \sigma_j  \rangle - \langle \hat{\sigma}_i \hat{\sigma}_j\rangle)^2

        Args:
            *plot (bool)*:
                Whether to plot the training schedule.

        Returns (dict):
            Dict with entries 'field' and 'coupling' with the respective MSE between the circuit and true statistics.

        """
        assert self.exact, "QVE needs to be called with argument exact=True to compare the statistics"
        # calculate expvals from groundstate wavefunction
        with tf.name_scope("exact_expvals"):
            gs = tf.reshape(tf.constant(self.gs_wavefunction, dtype=TF_COMPLEX_DTYPE),
                            [1] + [2 for _ in range(self.qc.nqubits)])
            expectation_layer = ExpectationValue(self.obs)
            exact_expvals = expectation_layer(gs)
        exact_expvals = self.qc._sess.run(flatten(exact_expvals))
        qc_expvals = self.qc._sess.run(flatten(self.expvals))
        mse = {}
        possible_fields = ['x', 'y', 'z']
        available_fields = [f for f in possible_fields if f in self.hamiltonian.interaction_slices.keys()]
        if available_fields:
            fig_fields, ax_fields = plt.subplots(2, len(available_fields))
            ax_fields = ax_fields.reshape((2, -1))
            pauli_f_names = dict(zip(possible_fields, [r"$\sigma^{}_i$".format(s[0]) for s in possible_fields]))
            fields_vals_qc = {}
            fields_vals_exact = {}
            for i, f in enumerate(available_fields):
                fields_vals_qc[f] = qc_expvals[
                                    self.hamiltonian.interaction_slices[f][0]:self.hamiltonian.interaction_slices[f][1]]
                fields_vals_exact[f] = exact_expvals[
                                       self.hamiltonian.interaction_slices[f][0]:self.hamiltonian.interaction_slices[f][
                                           1]]

                field_mse = np.mean((fields_vals_qc[f] - fields_vals_exact[f]) ** 2)
                mse['field'] = field_mse
                print("Field statistics {} MSE: {}".format(f, field_mse))
                if plot:
                    # make the figure for the fields
                    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                    stats = np.zeros((self.qc.nqubits))
                    for j, link in enumerate(self.hamiltonian.link_order[f]):
                        stats[link] = fields_vals_exact[f][j]
                    m = ax_fields[0, i].matshow(stats.reshape(1, -1), vmin=-1, vmax=1)
                    ax_fields[0, i].set_title('{}: exact'.format(pauli_f_names[f]))
                    ax_fields[0, i].grid(b=True, color='black', linewidth=2)
                    ax_fields[0, i].set_yticks([])
                    ax_fields[0, i].set_xticks([])
                    divider = make_axes_locatable(ax_fields[0, i])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig_fields.colorbar(m, cax=cax, orientation='vertical', norm=norm)

                    stats = np.zeros((self.qc.nqubits))
                    for j, link in enumerate(self.hamiltonian.link_order[f]):
                        stats[link] = fields_vals_qc[f][j]
                    m = ax_fields[1, i].matshow(stats.reshape(1, -1), vmin=-1, vmax=1)
                    ax_fields[1, i].set_title('{}: circuit'.format(pauli_f_names[f]))
                    ax_fields[1, i].grid(b=True, color='black', linewidth=2)
                    ax_fields[1, i].set_yticks([])
                    ax_fields[1, i].set_xticks([])
                    divider = make_axes_locatable(ax_fields[1, i])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig_fields.colorbar(m, cax=cax, orientation='vertical', norm=norm)
        # get the couplings
        possible_couplings = [''.join(f) for f in it.product(['x', 'y', 'z'], repeat=2)]
        available_couplings = [f for f in possible_couplings if f in self.hamiltonian.interaction_slices.keys()]
        if available_couplings:
            fig_couplings, ax_couplings = plt.subplots(2, len(available_couplings))
            ax_couplings = ax_couplings.reshape((2, -1))
            pauli_c_names = dict(
                zip(possible_couplings, [r"$\sigma^{}_i\sigma^{}_j$".format(s[0], s[1]) for s in possible_couplings]))
            coupling_vals_qc = {}
            coupling_vals_exact = {}
            for i, f in enumerate(available_couplings):
                coupling_vals_qc[f] = qc_expvals[
                                      self.hamiltonian.interaction_slices[f][0]:self.hamiltonian.interaction_slices[f][
                                          1]]
                coupling_vals_exact[f] = exact_expvals[self.hamiltonian.interaction_slices[f][0]:
                                                       self.hamiltonian.interaction_slices[f][1]]
                coupling_mse = np.mean((coupling_vals_qc[f] - coupling_vals_exact[f]) ** 2)
                mse['coupling'] = coupling_mse
                print("Coupling statistics {} MSE: {}".format(f, coupling_mse))
                if plot:
                    # make the figure for the couplings
                    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                    stats = np.zeros((self.qc.nqubits, self.qc.nqubits))
                    for j, link in enumerate(self.hamiltonian.link_order[f]):
                        stats[link[0], link[1]] = coupling_vals_exact[f][j]
                    m = ax_couplings[0, i].matshow(stats, vmin=-1, vmax=1)
                    ax_couplings[0, i].set_title('{}: exact'.format(pauli_c_names[f]))
                    ax_couplings[0, i].grid(b=True, color='black', linewidth=2)
                    ax_couplings[0, i].set_yticks([])
                    ax_couplings[0, i].set_xticks([])

                    divider = make_axes_locatable(ax_couplings[0, i])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig_couplings.colorbar(m, cax=cax, orientation='vertical', norm=norm)

                    # make the figure for the couplings
                    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                    stats = np.zeros((self.qc.nqubits, self.qc.nqubits))
                    for j, link in enumerate(self.hamiltonian.link_order[f]):
                        stats[link[0], link[1]] = coupling_vals_qc[f][j]
                    m = ax_couplings[1, i].matshow(stats, vmin=-1, vmax=1)
                    ax_couplings[1, i].set_title('{}: circuit'.format(pauli_c_names[f]))
                    ax_couplings[1, i].grid(b=True, color='black', linewidth=2)
                    ax_couplings[1, i].set_yticks([])
                    ax_couplings[1, i].set_xticks([])
                    divider = make_axes_locatable(ax_couplings[1, i])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig_couplings.colorbar(m, cax=cax, orientation='vertical', norm=norm)
        return mse

    def get_wavefunction(self, plot=False):
        r"""
        Compare the exact ground state wave function with the quantum circuit wave function by calculating the overlap

        .. math::

            R = |\langle \psi(\theta)| \psi_0\rangle|^2

        Args:
            plot (bool):
                Whether to plot the training schedule.

        Returns (float):
            The overlap :math:`R`

        """
        assert self.exact, "QVE needs to be called with argument exact=True to compare the statistics"
        if plot:
            raise NotImplementedError
        v_circuit = self.qc._sess.run(self.qc.execute())
        # small errors make it slightly larger than 1
        v_exact = self.gs_wavefunction.flatten()
        v_exact /= np.linalg.norm(v_exact)
        v_circuit = v_circuit.flatten()
        v_circuit /= np.linalg.norm(v_circuit)
        R1 = np.abs(np.sum(np.conjugate(v_circuit) * v_exact)) ** 2
        print("Overlap is {}".format(R1))
        return R1


class QuantumVariationalEigensolverGradFree(object):
    """
    Quantum Variational Eigensolver according to  `Peruzzo et al. (2014) <https://arxiv.org/abs/1304.3061>`_ using only gradient
    free optimizers.

    """

    def __init__(self, hamiltonian: Hamiltonian, qc: QuantumCircuit, exact=False, optimizer: str = None, **params):
        r"""
        We take a quantum circuit with parameters :math:`\theta` which
        implements some unitary transformation :math:`\mathcal{U}(\theta)`. This unitary brings our initial state
        :math:`|0\rangle` to a state :math:`|\psi(\theta)\rangle`.

        .. math::

            \mathcal{U}(\theta)|0\rangle = |\psi(\theta)\rangle

        Then, we realize that any Hamiltonian :math:`H` can be written as a polynomial of single qubit interactions.

        .. math::

            H = \sum_{i\alpha} h_\alpha^i \sigma_i^\alpha +  \sum_{ij\alpha\beta} w_{\alpha\beta}^{ij} \sigma_{i}^{\alpha}\sigma_{j}^{\beta} + \ldots

        where :math:`i,j,\ldots=1,\ldots,N` indicate the qubit and :math:`\alpha,\beta,\ldots=x,y,z` indicates the Pauli operator.
        By measuring the expectation values of the quantum circuit, we obtain a variational energy which is a lower bound on
        the true energy :math:`\langle H \rangle = \langle \psi| H |\psi \rangle`.

        .. math::

            \langle H \rangle \approx \langle H \rangle_{\theta^t} = \sum_{i\alpha} h_\alpha^i \langle\sigma_i^\alpha\rangle +
            \sum_{ij\alpha\beta} w_{\alpha\beta}^{ij} \langle \sigma_{i}^{\alpha} \sigma_{j}^{\beta}\rangle + \ldots

        By minimizing this variational energy, we hope to learn a quantum circuit which best represents the groundstate of
        :math:`H`. If this minimization is successful, we can easily sample the ground state.

        In the derivative-free method, we need only function evaluations at a point :math:`\theta^t`. In the case of the
        QVE this function is simply the expected energy under the variational circuit ansatz:

        .. math::

            f(\theta^t) = \langle H \rangle_{\theta^t} = \langle 0 | \mathcal{U}^\dagger(\theta^t) H  \mathcal{U}(\theta^t)|0\rangle

        Args:
            *hamiltonian (list)*:
                List of numpy arrays containing the operators and their locations. See :func:`~zyglrox.core.observables.verify_hamiltonian`
                for the required format.

            *qc (QuantumCircuit)*:
                ``QuantumCircuit`` object containing a quantum circuit architecture.

            *exact (bool)*:
                Boolean indicating whether we want to calculate the exact ground state and eigenvectors of the Hamiltonian.
                Will throw an error if the number of qubits is larger than 18.

            *device* (string):
                Device of choice for running tensorflow.

            *optimizer (str)*:
                Desired scipy optimizer, choose from 'Nelder-Mead', 'Powell', 'COBYLA', 'SLSQP', 'trust-constr' or 'trust-exact'.

        """
        assert isinstance(qc, QuantumCircuit), "qc must be a 'QuantumCircuit' object, received {}".format(type(qc))
        # assert verify_hamiltonian(hamiltonian)
        self.hamiltonian = hamiltonian
        # get all the non-zero fields from the hamiltonian
        self.obs = hamiltonian.get_observables()
        # set the hamiltonian terms
        self.hamiltonian_terms = hamiltonian.get_hamiltonian_terms()

        self.hamiltonian = hamiltonian
        self.qc = qc
        self.exact = exact
        self.TRAIN = False

        assert self.hamiltonian.nsites == self.qc.nqubits, "hamiltonian defined for {}, but quantum circuit has only {} qubits".format(
            self.hamiltonian.nsites, self.qc.nqubits)

        self.device = params.pop('device', "CPU")
        self.model_name = params.get("model_name", 'QVE')
        self.save_model = params.get("save_model", False)
        self.load_model = params.get("load_model", False)
        self.feed_in_hamiltonian_terms = params.get("feed_in_hamiltonian_terms", False)
        self.tfcheckpoint_path = params.get("tfcheckpoint_path", "./tfcheckpoints")
        if optimizer is None:
            self.optimizer = "Nelder-Mead"
        else:
            opt_list = ['Nelder-Mead', 'Powell', 'COBYLA', 'SLSQP', 'trust-constr', 'trust-exact']
            assert optimizer in opt_list, "Passed optimizer must be a gradient free optimizer implemented in scipy {}, received {}".format(
                opt_list, optimizer)
            self.optimizer = optimizer
        self._build_graph()

    def _build_graph(self):
        """
        Build the computational tensorflow graph.

        Args:

            device (string):
                Device of choice for running tensorflow.

        Returns (inplace):
            None

        """

        if self.exact:
            with tf.name_scope("exact_hamiltonian"):
                self.hamiltonian.get_hamiltonian()
                self.gs_energy = self.hamiltonian.energies[0]
                self.gs_wavefunction = self.hamiltonian.gs

        with tf.name_scope("scipy_handle"):
            self.theta_ph = tf.compat.v1.placeholder(dtype=TF_FLOAT_DTYPE, shape=(self.qc.nparams))
            c = 0
            for g in self.qc.gates:
                if g.nparams > 0:
                    g.set_external_input(tf.reshape(self.theta_ph[c:c + g.nparams], (1, g.nparams, 1)))
                    c += g.nparams

        phi = self.qc.circuit(self.qc.phi)

        with tf.name_scope("train_step"):
            self.expectation_layer = ExpectationValue(self.obs)
            self.expvals = self.expectation_layer(phi)
            if self.feed_in_hamiltonian_terms:
                self.hamiltonian_terms = tf.compat.v1.placeholder(dtype=TF_FLOAT_DTYPE, shape=len(self.hamiltonian_terms),
                                                                  name='terms')
                self.energy = tf.reduce_sum(flatten(self.hamiltonian_terms) * flatten(self.expvals), name="energy")
            else:
                self.energy = tf.reduce_sum(
                    tf.constant(self.hamiltonian_terms.flatten(), dtype=TF_FLOAT_DTYPE) * flatten(self.expvals),
                    name="energy")

        if self.save_model:
            self.saver = tf.compat.v1.train.Saver([self.qc.circuit.trainable_variables], max_to_keep=4)

        self.qc.initialize()

        if self.load_model:
            out = self.saver.restore(self.qc._sess, self.tfcheckpoint_path + self.model_name)
            print("Loaded model from {}".format(out))

    def scipy_fn(self, theta: np.ndarray):
        r"""
        Evaluate the energy for a given set of parameters :math:`\theta^t`.

        Args:
            theta (array):
                Array of size :math:`M` containing the parameters of the quantum circuit.

        Returns (float):
            Returns the energy :math:`\langle H \rangle_{\theta^t}` of the variational circuit.

        """
        energy = self.qc._sess.run(self.energy, feed_dict={self.theta_ph: theta})
        self.energy_per_epoch.append(energy)
        return energy

    def train(self, maxiter: int = 1000):
        r"""
        Train the quantum variational eigensolver. We minimize the energy :math:`\langle H \rangle_\theta` as defined above
        using a gradient free method.

        Args:
            *maxiter (int)*:
                The maximum number of iterations for the gradient free optimzer.

        Returns (inplace):
            None

        """
        self.TRAIN = True
        self.CONVERGED = False
        self.energy_per_epoch = []
        res = opt.minimize(fun=self.scipy_fn, x0=np.pi * (2 * np.random.rand(*(self.qc.nparams, 1)) - 1),
                           method=self.optimizer,
                           options={'adaptive': True, 'maxiter': maxiter})
        self.CONVERGED = res.success
        self.final_theta = res.x

    def get_convergence(self, plot=True) -> Union[float, None]:
        r"""
        Plot the energy at each epoch. If the argument `exact=True` was passed to the``QuantumVariationalEigensolverGradFree`` constructor we also
        calculates the residual energy :math:`\epsilon_{\text{res}} = |\text{min}(E_{qc} - E_{gs_wavefunction})|`. Otherwise returns None

        Args:
            *plot (bool)*:
                Wheter to plot the training schedule.

        Returns (float):
            Residual energy.

        """
        assert self.TRAIN, "Train the model before plotting"
        res = None
        grid = list(range(len(self.energy_per_epoch)))
        if plot:
            plt.plot(grid, np.array(self.energy_per_epoch).flatten(), label='QVE')
            if self.exact:
                res = np.abs(np.min(self.energy_per_epoch) - self.gs_energy)
                print("Residual Energy = {}".format(res))
                plt.ylim([self.gs_energy - 1, 0])
                plt.plot(grid, [self.gs_energy for _ in range(len(self.energy_per_epoch))], label='Exact')
            plt.xlabel('iteration')
            plt.ylabel(r'$E=\langle \psi_\theta|H|\psi_\theta\rangle$')
            plt.title("Convergence of Quantum Variational Eigensolver")
            plt.legend()

        return res

    def get_statistics(self, plot=True):
        r"""
        Compare the ground state statistics of the ``QuantumCircuit`` with the exact ground state statistics.

        .. math::

            \text{MSE}_{h} = \frac{1}{M} \sum_i^M (\langle \sigma_i \rangle - \langle \hat{\sigma}_i \rangle)^2 \\
            \text{MSE}_{w} = \frac{1}{M} \sum_{i,j}^M (\langle \sigma_i \sigma_j  \rangle - \langle \hat{\sigma}_i \hat{\sigma}_j\rangle)^2

        Args:
            *plot (bool)*:
                Whether to plot the training schedule.

        Returns (dict):
            Dict with entries 'field' and 'coupling' with the respective MSE between the circuit and true statistics.

        """
        assert self.exact, "QVE needs to be called with argument exact=True to compare the statistics"
        # calculate expvals from groundstate wavefunction
        with tf.name_scope("exact_expvals"):
            gs = tf.reshape(tf.constant(self.gs_wavefunction, dtype=TF_COMPLEX_DTYPE),
                            [1] + [2 for _ in range(self.qc.nqubits)])
            expectation_layer = ExpectationValue(self.obs)
            exact_expvals = expectation_layer(gs)
        exact_expvals = self.qc._sess.run(flatten(exact_expvals))
        qc_expvals = self.qc._sess.run(flatten(self.expvals), feed_dict={self.theta_ph: self.final_theta})

        mse = {}
        possible_fields = ['x', 'y', 'z']
        available_fields = [f for f in possible_fields if f in self.hamiltonian.interaction_slices.keys()]
        if available_fields:
            fig_fields, ax_fields = plt.subplots(2, len(available_fields))
            ax_fields = ax_fields.reshape((2, -1))
            pauli_f_names = dict(zip(possible_fields, [r"$\sigma^{}_i$".format(s[0]) for s in possible_fields]))
            fields_vals_qc = {}
            fields_vals_exact = {}
            for i, f in enumerate(available_fields):
                fields_vals_qc[f] = qc_expvals[
                                    self.hamiltonian.interaction_slices[f][0]:self.hamiltonian.interaction_slices[f][1]]
                fields_vals_exact[f] = exact_expvals[
                                       self.hamiltonian.interaction_slices[f][0]:self.hamiltonian.interaction_slices[f][
                                           1]]

                field_mse = np.mean((fields_vals_qc[f] - fields_vals_exact[f]) ** 2)
                mse['field'] = field_mse
                print("Field statistics {} MSE: {}".format(f, field_mse))
                if plot:
                    # make the figure for the fields
                    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                    stats = np.zeros((self.qc.nqubits))
                    for j, link in enumerate(self.hamiltonian.link_order[f]):
                        stats[link] = fields_vals_exact[f][j]
                    m = ax_fields[0, i].matshow(stats.reshape(1, -1), vmin=-1, vmax=1)
                    ax_fields[0, i].set_title('{}: exact'.format(pauli_f_names[f]))
                    ax_fields[0, i].grid(b=True, color='black', linewidth=2)
                    ax_fields[0, i].set_yticks([])
                    ax_fields[0, i].set_xticks([])
                    divider = make_axes_locatable(ax_fields[0, i])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig_fields.colorbar(m, cax=cax, orientation='vertical', norm=norm)

                    stats = np.zeros((self.qc.nqubits))
                    for j, link in enumerate(self.hamiltonian.link_order[f]):
                        stats[link] = fields_vals_qc[f][j]
                    m = ax_fields[1, i].matshow(stats.reshape(1, -1), vmin=-1, vmax=1)
                    ax_fields[1, i].set_title('{}: circuit'.format(pauli_f_names[f]))
                    ax_fields[1, i].grid(b=True, color='black', linewidth=2)
                    ax_fields[1, i].set_yticks([])
                    ax_fields[1, i].set_xticks([])
                    divider = make_axes_locatable(ax_fields[1, i])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig_fields.colorbar(m, cax=cax, orientation='vertical', norm=norm)
        # get the couplings
        possible_couplings = [''.join(f) for f in it.product(['x', 'y', 'z'], repeat=2)]
        available_couplings = [f for f in possible_couplings if f in self.hamiltonian.interaction_slices.keys()]
        if available_couplings:
            fig_couplings, ax_couplings = plt.subplots(2, len(available_couplings))
            ax_couplings = ax_couplings.reshape((2, -1))
            pauli_c_names = dict(
                zip(possible_couplings, [r"$\sigma^{}_i\sigma^{}_j$".format(s[0], s[1]) for s in possible_couplings]))
            coupling_vals_qc = {}
            coupling_vals_exact = {}
            for i, f in enumerate(available_couplings):
                coupling_vals_qc[f] = qc_expvals[
                                      self.hamiltonian.interaction_slices[f][0]:self.hamiltonian.interaction_slices[f][
                                          1]]
                coupling_vals_exact[f] = exact_expvals[self.hamiltonian.interaction_slices[f][0]:
                                                       self.hamiltonian.interaction_slices[f][1]]
                coupling_mse = np.mean((coupling_vals_qc[f] - coupling_vals_exact[f]) ** 2)
                mse['coupling'] = coupling_mse
                print("Coupling statistics {} MSE: {}".format(f, coupling_mse))
                if plot:
                    # make the figure for the couplings
                    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                    stats = np.zeros((self.qc.nqubits, self.qc.nqubits))
                    for j, link in enumerate(self.hamiltonian.link_order[f]):
                        stats[link[0], link[1]] = coupling_vals_exact[f][j]
                    m = ax_couplings[0, i].matshow(stats, vmin=-1, vmax=1)
                    ax_couplings[0, i].set_title('{}: exact'.format(pauli_c_names[f]))
                    ax_couplings[0, i].grid(b=True, color='black', linewidth=2)
                    ax_couplings[0, i].set_yticks([])
                    ax_couplings[0, i].set_xticks([])

                    divider = make_axes_locatable(ax_couplings[0, i])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig_couplings.colorbar(m, cax=cax, orientation='vertical', norm=norm)

                    # make the figure for the couplings
                    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                    stats = np.zeros((self.qc.nqubits, self.qc.nqubits))
                    for j, link in enumerate(self.hamiltonian.link_order[f]):
                        stats[link[0], link[1]] = coupling_vals_qc[f][j]
                    m = ax_couplings[1, i].matshow(stats, vmin=-1, vmax=1)
                    ax_couplings[1, i].set_title('{}: circuit'.format(pauli_c_names[f]))
                    ax_couplings[1, i].grid(b=True, color='black', linewidth=2)
                    ax_couplings[1, i].set_yticks([])
                    ax_couplings[1, i].set_xticks([])
                    divider = make_axes_locatable(ax_couplings[1, i])
                    cax = divider.append_axes('right', size='5%', pad=0.05)
                    fig_couplings.colorbar(m, cax=cax, orientation='vertical', norm=norm)
        return mse

    def get_wavefunction(self, plot=False):
        r"""
        Compare the exact ground state wave function with the quantum circuit wave function by calculating the overlap

        .. math::

            R = |\langle \psi(\theta)| \psi_0\rangle|^2

        Args:
            plot (bool):
                Whether to plot the training schedule.

        Returns (float):
            The overlap :math:`R`

        """
        assert self.exact, "QVE needs to be called with argument exact=True to compare the statistics"
        if plot:
            raise NotImplementedError
        v_circuit = self.qc._sess.run(self.qc.execute(), feed_dict={self.theta_ph: self.final_theta})
        # small errors make it slightly larger than 1
        v_exact = self.gs_wavefunction.flatten()
        v_exact /= np.linalg.norm(v_exact)
        v_circuit = v_circuit.flatten()
        v_circuit /= np.linalg.norm(v_circuit)
        R1 = np.abs(np.sum(np.conjugate(v_circuit) * v_exact)) ** 2
        print("Overlap is {}".format(R1))
        return R1
