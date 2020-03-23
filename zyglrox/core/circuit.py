import tensorflow as tf
from tensorflow.keras.layers import Layer

import numpy as np
from typing import List
from zyglrox.core._config import TF_COMPLEX_DTYPE
from zyglrox.core.gates import Gate
from zyglrox.core.utils import integer_generator


class QuantumCircuit:
    """
    Quantum Circuit TensorFlow Interface
    """

    def __init__(self, nqubits: int, gates: List, tensorboard=False, **kwargs):
        """
        This class is an interface to build quantum circuits using TensorFlow. On initialization, the graph for the circuit is constructed.
        The circuit is defined by passing a list of ``Gate`` objects, and setting the measurement of observables through the method ``set_expval``.

        Args:
            *nqubits (int)*:
                Number of qubits.

            *gates (list)*:
                List of ``Gate`` objects.

            *tensorboard (bool)*:
                Boolean that indicates whether we store the store meta data for tensorboard. Default false.

            :Keyword Arguments:
                *batch_size (int)*:
                    Number of parallel iterations for map_fn when using batches of parammeters.

        Returns (inplace):
            None

        """
        self.nqubits = nqubits
        assert isinstance(gates, List), "gates must be a list, received {}".format(type(gates))
        assert len(gates) != 0, "gates is an empty list"
        assert all([isinstance(el, Gate) for el in gates]), "gates must contain only 'Gate' objects"
        self.gates = gates
        self.tensorboard = tensorboard
        self.batch_params = kwargs.pop('batch_params', False)
        self.batch_size = kwargs.pop('batch_size', 16)
        self.nparams = 0
        self.ngates = 0
        self.nlayers = 0
        self.device = kwargs.pop('device', 'CPU')
        self.circuit_order = kwargs.pop('circuit_order', 'gate')
        self.get_phi_per_layer = kwargs.pop('get_phi_per_layer', False)
        assert self.circuit_order in ['gate', 'layer']
        self.device_number = kwargs.pop('device_number', 0)
        for g in gates:
            if g.nparams == 0:
                self.ngates += 1
            else:
                self.ngates += g.nparams
            self.nparams += g.nparams
        self.build_graph()

    def __str__(self):
        return "Quantum Circuit\n" + \
               "---------------\n" + \
               "Number of qubits: {}\n".format(self.nqubits) + \
               "Number of gates: {}\n".format(self.ngates) + \
               "Number of parameters: {}\n".format(self.nparams)

    def build_graph(self):
        """
        Build the computational graph of the quantum circuit.

        Returns (inplace):
            None

        """

        with tf.name_scope("phi"):
            # initalize the zero quantum state with a constant tensor
            self.phi = np.zeros((int(2 ** self.nqubits)))
            self.phi[0] = 1
            self.phi = tf.constant(self.phi, dtype=TF_COMPLEX_DTYPE, name='phi')
            # reshape to multi-index form
            self.phi = tf.reshape(self.phi, [1] + [2 for _ in range(self.nqubits)])
        with tf.name_scope("circuit"):
            # Set the first gate to be the input layer. This will be useful later.
            self.gates[0]._input_layer = True
            if self.batch_params:
                for g in self.gates:
                    g._batch_params = True
                    g._batch_size = self.batch_size

            if self.circuit_order == 'gate':
                self.circuit = tf.keras.Sequential(layers=self.gates, name='circuit')
            elif self.circuit_order == 'layer':
                self._get_layer_ordering()
                layers = []
                for l in range(self.nlayers):
                    layers.append(CircuitLayer(gates=self.gates_per_layers[l], name='layer_{}'.format(l)))

                if self.get_phi_per_layer:
                    assert not self.batch_params, "batch_params is not supported when getting the wave functions per layer"

                    layers = [tf.keras.layers.Input(tensor=self.phi)] + layers
                    self.circuit = tf.keras.Sequential(layers=layers, name='circuit')
                    phi_per_layer = []

                    for l in range(self.nlayers):
                        phi_per_layer.append(self.circuit.layers[l].output)
                    self.phi_per_layer = tf.concat(phi_per_layer, axis=0)

                else:
                    self.circuit = tf.keras.Sequential(layers=layers, name='circuit')

    def initialize(self) -> None:
        """
        Initialize session, variables, and the Tensorboard writer.

        Args:

            *device* (string):
                Device of choice for running tensorflow.

        Returns (inplace):
            None

        """

        assert self.device in ['CPU', 'GPU'], "device must be one of '['CPU', 'GPU'], received {}".format(self.device)
        assert isinstance(self.device_number, int), 'device_number must be and integer, received {}'.format(
            type(self.device_number))

        with tf.device("/" + self.device + ":" + str(self.device_number)):
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            self._sess = tf.compat.v1.Session(config=config)

        self._sess.run([tf.compat.v1.global_variables_initializer()])

        if self.tensorboard:
            self.writer = tf.compat.v1.summary.FileWriter(logdir='logdir', graph=self._sess.graph)

    def execute(self) -> tf.Tensor:
        r"""
        Exectute the circuit, starting from the initial zero state :math:`|0\rangle^{\otimes N}` .

        Returns (Tensor):

            Circuit output, a wave function of shape (None, 2,...,2).

        """
        return self.call(self.phi)

    def call(self, inputs, training=False) -> tf.Tensor:
        r"""
        Call the circuit for a given wave function or batch of wave functions

        Args:
            *inputs (Tensor)*:
                Input tensor corresponding to the wave function of shape (None, 2,...,2).
                In the case of a single wave function we get a shape (1,2,...,2)

            *training (Bool)*:
                Indicates whether to run the circuit in training mode or inference mode.

        Returns (Tensor):

            Circuit output, a wave function of shape (None, 2,...,2).

        """

        return self.circuit(inputs, training)

    def set_parameters(self, theta):
        """
        Set the parameters of the quantum circuit by an external input. Also works for batches of parameters
        if the QuantumCircuit has been initialized with batch_params=True.

        Args:
            *theta (tensor)*:
                Parameters to be fed into the circuit, either of dimension (1, nparams, 1) or (None,nparams,1) where
                nparams is the total number of parameters in the circuit.

        Returns:

        """
        theta_size = theta.shape.as_list()
        assert len(theta_size) == 3, "expected tensor with 3 dimensions, received tensor with shape {}".format(
            theta_size)
        assert theta.shape.as_list()[
                   1] == self.nparams, "Expected dim(2) of theta to be equal to the total number of parameters " \
                                       "in the circuit; {} but received {}".format(self.nparams, theta_size)
        with tf.name_scope("circuit_handle"):
            i = 0
            for g in self.gates:
                if g.nparams > 0:
                    g.set_external_input(tf.reshape(theta[:, i:i + g.nparams], (-1, g.nparams, 1)))
                    i += g.nparams
        assert i == self.nparams, "Done setting i parameters, but we have {} parameterized gates".format(self.nparams)

    def _get_layer_ordering(self):
        assert self.circuit_order == 'layer', "Wrong circuit_type: {}".format(self.circuit_order)

        layers = {}
        self.gates_per_layers = {}

        s = 0
        for g in self.gates:
            for l in integer_generator(s):
                if l not in layers.keys():
                    layers[l] = set(range(1, self.nqubits + 1))
                if all([w in layers[l] for w in g.wires]):
                    g.layer = l
                    for w in g.wires:
                        layers[l].remove(w)
                    break
            if len(layers[s]) == 0:
                s += 1

        for g in self.gates:
            if g.layer not in self.gates_per_layers.keys():
                self.gates_per_layers[g.layer] = []
            self.gates_per_layers[g.layer].append(g)
            
        self.nlayers = len(self.gates_per_layers)


class CircuitLayer(Layer):
    """
    Abstract Quantum Gate class

    """

    def __init__(self, gates, name=None, **kwargs):
        """
        The ``CircuitLayer`` combines a set of commuting gates into a single Tensorflow layer

        Args:
            *nparams (int)*:
                Number of parameters of gate.

            *wires (list)*:
                List of numbers on which the gate is acting.

            *value (None of ndarray)*:
                Intial value of parameterized gate in 1D numpy array. When combined with setting ``trainable=False``, this
                will create a static gate that cannot be altered during any optimization.

            *name (str)*:
                Name of the gate.

            :Keyword Arguments:

                *trainable (bool)*:
                    Boolean that indicates whether the paramaters of this gate are trainable.

        Returns (inplace):
            None

        """
        allowed_kwargs = ['trainable']
        checked_kwargs = [k in allowed_kwargs for k in kwargs.keys()]
        assert all(checked_kwargs), "{} is not an allowed keyword argument".format(
            next(list(kwargs.keys())[i] for i in range(len(checked_kwargs)) if not checked_kwargs[i]))

        super(CircuitLayer, self).__init__(name=name, **kwargs)
        layers = [g.layer for g in gates]
        assert len(set(layers))==1, "Layer attributes of all passed gates must be equal, received {}".format(layers)
        assert all([isinstance(g, Gate) for g in gates]), "Layer attributes of all passed gates must be equal, received {}".format(layers)
        self.gates = gates
        self.layer = gates[0].layer

    def __str__(self):
        return self.name

    def build(self, input_shape):
        """
        Called once from `__call__`, when we know the shapes of inputs and `dtype`.
        Should initialize the trainable variables, and call the super's `build()`.

        Args:
            *input_shape (list)*:
                Input shapes of the incoming tensor.

        """
        super(CircuitLayer, self).build(input_shape)
        self.built = True


    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic goes here.

        """
        phi = self.gates[0](inputs)
        for i in range(1,len(self.gates)):
            phi = self.gates[i](phi)
        phi = tf.identity(phi, name='phi_layer_{}'.format(self.layer))
        return phi