********
Beginner
********

##################
1. Getting Started
##################

In this tutorial we will explore the basic workflow of using ``zyglrox``.
We start by defining a quantum circuit. First, we import the necessary modules:
the ``QuantumCircuit`` class, two gates and the ``Observable`` class.

.. code-block:: python

	from zyglrox.core.circuit import QuantumCircuit
	from zyglrox.core.gates import Hadamard, PauliZ
	from zyglrox.core.observables import Observable

Next, we will will construct a quantum circuit on a single qubit, where we first apply a Hadmard gate, and afterwards a Pauli Z gate.
Specifiying the circuit is done by make a list of ``Gate`` objects, where for each gate we specify on which 'quantum wire' they operate.

.. code-block:: python

	gates = [Hadamard(wires=[0,]), PauliZ(wires=[0,])]
	qc = QuantumCircuit(nqubits=1, gates=gates, device="CPU")

Before we can calculate anything, we need to initialize the model in true TensorFlow fashion. If you have a CUDA enabled GPU available,
you can choose to run the calculation on the GPU. For now, we will select the CPU to do the heavy lifting.

.. code-block:: python

	qc.initialize()

.. note::

	``zyglrox`` relies on ``tensorflow 1.15.0``. This Python framework uses an imperative programming paradigm to
	construct a computational graph, which is then executed in a so called 'session'.

	.. image:: ../_static/png/tut1_tf_graph.png
		:align: center

	Understanding and managing these graphs and sessions requires some understanding of TensorFlow. However, when using ``zyglrox`` you only have to worry
	about calling ``qc.initalize`` after all the circuit definitions are done.

Now we are ready to do some calculations. ``QuantumCircuit`` has a method ``circuit``, that can be called  on a complex vector to output
the wavefunction after applying the gates. Additionally, ``QuantumCircuit`` contains a constant vector ``qc.phi`` corresponding to the :math:`|0\rangle^{\otimes N}`,
that can be propagated through the circuit. We can call the initialized session to extract the values of interest from the graph.

.. code-block:: python

	phi = qc.circuit(qc.phi)
	# graph definitions are done
	qc_tf = qc._sess.run(phi)
	print(qc_tf)
	>>> [ 0.70710677+0.j -0.70710677+0.j]

Which outputs the correct state:

.. math::

	\mathcal{U}|{0}\rangle = Z \: H |{0}\rangle \equiv
	\begin{pmatrix}
	1 &  0\\
	0 & -1
	\end{pmatrix}
	\frac{1}{\sqrt{2}}
	\begin{pmatrix}
	1 &  1\\
	1 & -1
	\end{pmatrix}
	\begin{pmatrix}
	1 \\
	0
	\end{pmatrix}
	=
	\frac{1}{\sqrt{2}}
	\begin{pmatrix}
	1 \\
	-1
	\end{pmatrix}

On a real quantum computer, we do not have access to the full wave function. Instead, we have to rely on the measurement of
observables to obtain information about the state of the system. To simulate this, we add an observable layer to our quantum circuit.
For instance, lets say we want to measure in the X-basis on the first qubit.

.. code-block:: python

	obs = [Observable("x", wires=[0, ])]
	expval_layer = ExpectationValue(obs)

This adds the ``ExpectationValue`` layer to the graph, a layer containing all the observables of interest measured on our quantum circuit.
To retrieve the observable :math:`\langle \sigma^x \rangle` from the graph, we run the ``ExpectationValue`` layer and extract the observables.

.. code-block:: python

	measurements = qc._sess.run(expval_layer(phi))
	print(measurements)
	>>> [[[[-0.99999994]]]]

Which is again the value we expect.

######################################
2. Multi-Qubit gates and Visualization
######################################

For this tutorial we will explore the usage of multi-qubit gates and parametrizable gates. First we import the ``QuantumCircuit``, some gates
and the ``Observable`` class.

.. code-block:: python

	from zyglrox.core.circuit import QuantumCircuit
	from zyglrox.core.gates import Hadamard, Phase, CNOT
	from zyglrox.core.observables import Observable
	import numpy as np

Next, we will define the gates of our circuit for 2 qubits. The ``Phase`` gate rotates the qubit state around the z-axis of the Bloch sphere

.. math::

	R(\theta) =
	\begin{pmatrix}
	1 & 0\\
	0 & e^{i\theta}
	\end{pmatrix}

where :math:`\theta` is the so-called phase shift. When we move on to variational circuits, this variable will be adjustable. For
now, we set this parameter to :math:`\pi/8`.

In order to make use of entanglement, we need integrate CNOT gates, since these
gates turn a product state

.. math::

	|\psi \rangle = (\alpha_1 |0\rangle + \beta_1 |1\rangle )\otimes(\alpha_2 |0\rangle  + \beta_2 |1\rangle )

into a linear combination of pure states

.. math::

	|\psi \rangle = \alpha_1 |0\rangle \otimes(\alpha_2 |0\rangle  + \beta_2 |1\rangle ) + \beta_1 |1\rangle \otimes(\beta_2 |0\rangle  + \alpha_2 |1\rangle )

by conditionally flipping the target qubit (in this case the second qubit) if the first qubit is in the :math:`|1\rangle` state.
We will apply a single CNOT at the end of our circuit.

In ``zyglrox``, we define this circuit as follows

.. code-block:: python

	gates = [Hadamard(wires=[0, ]), Phase(wires=[0, ], value=[np.pi / 8]),
		 Hadamard(wires=[1, ]), Phase(wires=[1, ], value=[np.pi / 8]), CNOT(wires=[0, 1])]

To enable visualization in TensorBoard, we pass the ``tensorboard=True`` argument to the ``QuantumCircuit`` constructor.

.. code-block:: python

	qc = QuantumCircuit(nqubits=2, gates=gates,tensorboard=True)

Since the ``qc.circuit`` method is a sequential Keras model, we can call the ``summary()`` function on this object to print
the parameters and layers of this circuit.

.. code-block:: python

	phi = qc.circuit(qc.phi)
	qc.circuit.summary()
	>>> Model: "circuit"
	_________________________________________________________________
	Layer (type)                 Output Shape              Param #
	=================================================================
	H_0 (Hadamard)               multiple                  0
	_________________________________________________________________
	Phase_0 (Phase)              multiple                  1
	_________________________________________________________________
	CNOT_0_1 (CNOT)              multiple                  0
	_________________________________________________________________
	H_1 (Hadamard)               multiple                  0
	_________________________________________________________________
	Phase_1 (Phase)              multiple                  1
	_________________________________________________________________
	CNOT_1_0 (CNOT)              multiple                  0
	=================================================================
	Total params: 2
	Trainable params: 2
	Non-trainable params: 0

.. note::

	Printing out the commonly used quantum circuit representation of wires and blocks of gates will be added in a future release.

In ``zyglrox``, we are not limited by the physical constraints of a quantum computer. We can extract multiple
observables in parallel, even from the same qubit.

.. code-block:: python

	obs = [Observable("x", wires=[0, ]), Observable("x", wires=[1, ]),
		Observable("y", wires=[0, ]), Observable("y", wires=[1, ]),
		Observable("z", wires=[0, ]), Observable("z", wires=[1, ])]
	expval_layer = ExpectationValue(obs)

Now that we're done, we initialize the session and extract the measurements.

.. code-block:: python

	qc.initialize()
	measurements = qc._sess.run(expval_layer(phi))
	print(measurements)
	>>> [[[[ 8.5355318e-01]]
        [[ 9.2387938e-01]]
        [[ 3.5355335e-01]]
        [[-7.4505806e-09]]
        [[ 2.9802322e-08]]
        [[ 1.4901161e-08]]]]

Additionally, we can visualize the computational graph in TensorBoard. When enabled, the logs are automatically stored in the *./logdir* folder.

.. code-block:: bash

	>>> tensorboard --logdir=logdir

which looks like |tut2_tensorboard|.

.. |tut2_tensorboard| raw:: html

		<a href="../_static/png/tut2_tensorboard.png" target="_blank"> this.
		</a>

************
Intermediate
************

################
1. Making Things
################

List comprehendsions to make advanced templates

##############
2. Doing Stuff
##############

Gradients

************
Advanced
************

##################
1. Learning Magic
##################

###################
2. Practicing Magic
###################
