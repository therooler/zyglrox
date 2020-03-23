Core
============
The core module contains the most important code parts used in ``zyglrox``

Circuit
-------------------
The ``Circuit`` class contains all functionality for constructing quantum circuits and calculating observables. The circuit
is defined by passing a list of ``Gate`` objects to the constructor. The combination of :math:`M` quantum gates is equal to a single unitary operation

.. math::

	\mathcal{U} \equiv \prod_i^M \mathcal{U}_i

Our quantum computer then calculates the state

.. math::

	|\psi\rangle = \mathcal{U} |0\rangle

If the gates :math:`\mathcal{U}_i` are paramterized by some parameter :math:`\theta_i`, so :math:`\mathcal{U}_i\to\mathcal{U}_i(\theta_i)`,
we have what is called a variational quantum circuit

.. math::

	|\psi(\theta)\rangle = \mathcal{U}(\theta) |0\rangle

These variational quantum circuits are essential for algorithms such as the Quantum Variational Eigensolver, the Quantum Boltzmann Machine or Quantum Approximate Adiabatic Optimization.
For more information on the available gates in ``zyglrox``, see the section
about :ref:`gates_section`.

.. note::

	This class inherits from the ``tf.keras.models.Model`` class, and can thus be accessed in a similar fashion as other Keras
	Models. This allows for easy integration of your variational quantum circuit with the deep learning tools available in TensorFlow and Keras.

.. autoclass:: zyglrox.core.circuit.QuantumCircuit(tf.keras.models.Model)
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:


.. _gates_section:

Gates
-----------------
A quantum gate acting :math:`\mathcal{U}` on a single qubit :math:`i`, acts on full the tensor subspace of the :math:`N`-qubit state vector.
To perform this calculation, we can calculate the tensor representation of this operation on the full state space as follows:

.. math::

   \mathcal{U}_N = \otimes^{i-1}_{j=0} I \otimes \mathcal{U} \otimes^{N}_{j=i+1} I

However, this provides a lot of overhead, since for every gate we need to calculate and store a :math:`2^N\times2^N` matrix.

A more efficient method is to only let the quantum gate act on the relevant tensor subspace. Throughout ``zyglrox``,
the wave function :math:`|\psi\rangle^{\otimes N}` is stored as a tensor of shape :math:`(2,2,\ldots,2)`. The gate operation can
then be performed as:

.. math::

   |\hat{\psi}\rangle = \otimes^{i-1}_{j=0} |\psi_j\rangle \otimes \mathcal{U} |\psi_i\rangle \otimes^{N}_{j=i+1} |\psi_j\rangle

Which can be implemented in ``tensorflow`` in the following way:

.. code-block:: python

   phi_hat = tf.tensordot(self.op, phi,
                          axes=[list(range(self.nqubits, 2 * self.nqubits)), self.wires]
                          )
   # tensordot sets the contracted axes first, so we need to reshuffle them
   unused_idxs = [idx for idx in range(self.total_qubits) if idx not in self.wires]
   perm = self.wires + unused_idxs
   inv_perm = np.argsort(perm)
   tf.transpose(phi, perm=inv_perm)

.. currentmodule:: zyglrox.core

.. autoclass:: zyglrox.core.gates.Gate

--------------
Gate Templates
--------------

Below is a table of the most commonly used quantum gates that are already implemented in ``zyglrox``.

.. autosummary::
   :toctree:

   gates.CNOT
   gates.Hadamard
   gates.PauliX
   gates.PauliY
   gates.PauliZ
   gates.RX
   gates.RY
   gates.RZ
   gates.R3
   gates.CRX
   gates.CRY
   gates.CRZ
   gates.CR3
   gates.CZ
   gates.Phase
   gates.XX
   gates.YY
   gates.ZZ
   gates.Swap
   gates.Toffoli


Observables
-----------------------
We can calculate observables :math:`\mathcal{O}` in a similar fashion as quantum gates.
By letting the observable of interest work on the required tensor subspace, we can efficiently obtain the corresponding expectation value.

.. math::

   \langle \psi| \mathcal{O} |\psi \rangle = \langle \psi |\hat{\psi}\rangle

where :math:`|\hat{\psi}\rangle` is calculated as in the section about :ref:`gates_section`


.. automodule:: zyglrox.core.observables
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

Hamiltonians
---------------------

.. currentmodule:: zyglrox.core

.. autoclass:: zyglrox.core.hamiltonians.Hamiltonian

---------------------
Hamiltonian Templates
---------------------

Below is a table of the most commonly used quantum gates that are already implemented in ``zyglrox``.

.. autosummary::
   :toctree:

   hamiltonians.TFI
   hamiltonians.HeisenbergXXX
   hamiltonians.HeisenbergXXZ
   hamiltonians.HeisenbergXYZ
   hamiltonians.RandomFullyConnectedXYZ
   hamiltonians.J1J2


Circuit Templates
---------------------

.. automodule:: zyglrox.core.circuit_templates
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

Topologies
---------------------

.. automodule:: zyglrox.core.topologies
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

Edge Coloring
---------------------

.. automodule:: zyglrox.core.edge_coloring
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:

Utils
-----------------

.. automodule:: zyglrox.core.utils
   :members:
   :member-order: bysource
   :undoc-members:
   :show-inheritance:
