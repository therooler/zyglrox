from zyglrox.models.qve import QuantumVariationalEigensolver
from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.hamiltonians import Hamiltonian
from zyglrox.core.observables import ExpectationValue
from zyglrox.core.utils import flatten

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

import itertools as it


class QuantumBoltzmannMachine():
    """
    Quantum Boltzmann Machine according to `Kappen (2019) <https://arxiv.org/abs/1803.11278>`_

    """

    def __init__(self, nqubits: int, qc: QuantumCircuit, pdf: np.ndarray, hamiltonian, optimizer=None):
        r"""
        Initialize

        Args:
            nqubits (int):
                The number of qubits in the system.

            qc (QuantumCircuit):
                Parametrized quantum circuit for :math:`N` spins.

            pdf (np.ndarray):
                array containing the target distribution :math:`q(x)`.

            optimizer (tf.optimizer.Optimizer):
                Tensorflow optimizer.

        Returns (inplace):
            None

        """
        self.nspins = nqubits
        self.qc = qc
        self.qve = QuantumVariationalEigensolver(hamiltonian, qc, device="GPU", optimizer=optimizer, exact=True, feed_in_hamiltonian_terms=True)

        with tf.name_scope("target_expvals"):
            target_phi = tf.cast(tf.reshape(tf.stack(np.sqrt(pdf)), [1] + [2 for _ in range(self.nspins)]),
                                 dtype=tf.complex64)
            expval_layer = ExpectationValue(self.qve.obs)
            target_expvals = expval_layer(target_phi)
        self.target_expvals = self.qve.qc._sess.run(target_expvals).flatten()
        self.TRAIN = False


    def train(self, eta_qbm=1e-3, tol_qbm=1e-4, tol_qve=1e-8, epochs_qbm=100,
              epochs_qve=1500):
        r"""
        Train the quantum Boltzmann machine. We minimize the value of the gradient :math:`\frac{1}{M}\sum_i^M |\nabla_{\theta_i} \mathcal{L}(\theta)|`
        as defined above

        Args:
            eta_qbm (float):
                Learning rate for the ``QuantumBoltzmannMachine``.

            tol_qbm (float):
                Tolerance :math:`\epsilon_{qbm}` on the mean squared error of the statistics. If the absolute difference between iterations
                is smaller than this value, training stops. ``QuantumBoltzmannMachine``.

            tol_qve (float):
                Tolerance :math:`\epsilon_{qve}` on the energy. If the absolute difference between iterations is smaller than this value, training stops.

            epochs_qbm (int):
                Number of max iterations for the training algorithm of the ``QuantumBoltzmannMachine``.

            epochs_qve (int):
                Number of max iterations for the training algorithm ``QuantumVariationalEigensolver``.

        Returns (inplace):
                    None

        """
        self.total_grad_per_epoch = []
        self.TRAIN = True
        for i in range(epochs_qbm):
            print("\nQBM Epoch {}".format(i))
            print("-------------------------")
            evals = self.qve.train(tol=tol_qve, epochs=epochs_qve)
            evals = evals.flatten()
            total_grad = 0
            j=0
            for term in self.qve.hamiltonian.interaction_order:
                for link in self.qve.hamiltonian.link_order[term]:
                    grad = self.target_expvals[j] - evals[j]
                    total_grad += np.abs(grad)
                    self.qve.hamiltonian.model_parameters[term][tuple(link)] -= eta_qbm * grad
                    j+=1
            total_grad /= j
            self.total_grad_per_epoch.append(total_grad)
            print("Absolute value of gradient: {}".format(total_grad))
            if total_grad < tol_qbm:
                self.qve.hamiltonian.get_hamiltonian_terms()
                break
            if i == epochs_qbm - 1:
                break
            self.qve.hamiltonian_terms = self.qve.hamiltonian.get_hamiltonian_terms()
            self.qc._sess.run([tf.compat.v1.global_variables_initializer()])

    def plot(self):
        r"""

        Plot the convergence of the ``QuantumBoltzmannMachine`` learning step. Convergence is obtained when
        :math:`\frac{1}{M}\sum_i^M |\nabla_{\theta_i} \mathcal{L}(\theta)|<\epsilon_{qbm}`, where :math:`\epsilon_{qbm}` is the tolerance
        defined when calling the ``train`` method.

        Returns (inplace):
            None

        """
        assert self.TRAIN, "Train the model before plotting"
        grid = list(range(len(self.total_grad_per_epoch)))
        plt.plot(grid, self.total_grad_per_epoch)
        plt.xlabel("Epoch")
        plt.ylabel(r"$\frac{1}{M}\sum_i^M |\nabla_{\theta_i} \mathcal{L}(\theta)|$")
        plt.yscale('log')
        plt.ylim([min(self.total_grad_per_epoch) * 0.95, max(self.total_grad_per_epoch) * 1.1])
        plt.title("Convergence of Quantum Boltzmann Machine for n={}".format(self.nspins))

    def get_statistics(self, plot=True):
        r"""
        Compare the statistics of learned ``QuantumBoltzmannMachine`` with the target ground state statistics.

        .. math::

            \text{MSE}_{h} = \frac{1}{M} \sum_i^M (\langle \sigma_i \rangle - \langle \hat{\sigma}_i \rangle)^2 \\
            \text{MSE}_{w} = \frac{1}{M} \sum_{i,j}^M (\langle \sigma_i \sigma_j  \rangle - \langle \hat{\sigma}_i \hat{\sigma}_j\rangle)^2

        .. warning:: We do not for sure know if the ``QuantumVariatonalEigensolver`` was succesful in obtaining the ground state of the model
         QBM hamiltonian unless we compare it with the exact ground state first.

        Args:
            plot (bool):
                Whether to plot the training schedule.

        Returns (dict):
            Dict with entries 'field' and 'coupling' with the respective MSE between the circuit and true statistics.

        """
        assert self.TRAIN, "Train the model before plotting"
        assert self.qve.exact, "QVE needs to be called with argument exact=True to compare the statistics"

        qc_expvals = self.qc._sess.run(flatten(self.qve.expvals))
        mse = {}
        possible_fields = ['x', 'y', 'z']
        available_fields = [f for f in possible_fields if f in self.qve.hamiltonian.interaction_slices.keys()]
        if available_fields:
            fig_fields, ax_fields = plt.subplots(2, len(available_fields))
            ax_fields = ax_fields.reshape((2, -1))
            pauli_f_names = dict(zip(possible_fields, [r"$\sigma^{}_i$".format(s[0]) for s in possible_fields]))
            fields_vals_qc = {}
            fields_vals_exact = {}
            for i, f in enumerate(available_fields):
                fields_vals_qc[f] = qc_expvals[
                                    self.qve.hamiltonian.interaction_slices[f][0]:self.qve.hamiltonian.interaction_slices[f][1]]
                fields_vals_exact[f] = self.target_expvals[
                                       self.qve.hamiltonian.interaction_slices[f][0]:self.qve.hamiltonian.interaction_slices[f][
                                           1]]

                field_mse = np.mean((fields_vals_qc[f] - fields_vals_exact[f]) ** 2)
                mse['field'] = field_mse
                print("Field statistics {} MSE: {}".format(f, field_mse))
                if plot:
                    # make the figure for the fields
                    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                    stats = np.zeros((self.qc.nqubits))
                    for j, link in enumerate(self.qve.hamiltonian.link_order[f]):
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
                    for j, link in enumerate(self.qve.hamiltonian.link_order[f]):
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
        available_couplings = [f for f in possible_couplings if f in self.qve.hamiltonian.interaction_slices.keys()]
        if available_couplings:
            fig_couplings, ax_couplings = plt.subplots(2, len(available_couplings))
            ax_couplings = ax_couplings.reshape((2, -1))
            pauli_c_names = dict(
                zip(possible_couplings, [r"$\sigma^{}_i\sigma^{}_j$".format(s[0], s[1]) for s in possible_couplings]))
            coupling_vals_qc = {}
            coupling_vals_exact = {}
            for i, f in enumerate(available_couplings):
                coupling_vals_qc[f] = qc_expvals[
                                      self.qve.hamiltonian.interaction_slices[f][0]:self.qve.hamiltonian.interaction_slices[f][
                                          1]]
                coupling_vals_exact[f] = self.target_expvals[self.qve.hamiltonian.interaction_slices[f][0]:
                                                       self.qve.hamiltonian.interaction_slices[f][1]]
                coupling_mse = np.mean((coupling_vals_qc[f] - coupling_vals_exact[f]) ** 2)
                mse['coupling'] = coupling_mse
                print("Coupling statistics {} MSE: {}".format(f, coupling_mse))
                if plot:
                    # make the figure for the couplings
                    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
                    stats = np.zeros((self.qc.nqubits, self.qc.nqubits))
                    for j, link in enumerate(self.qve.hamiltonian.link_order[f]):
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
                    for j, link in enumerate(self.qve.hamiltonian.link_order[f]):
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
