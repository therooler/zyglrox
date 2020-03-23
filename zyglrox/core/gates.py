import tensorflow as tf
from zyglrox.core.utils import tf_kron, tensordot
from zyglrox.core._config import TF_FLOAT_DTYPE, TF_COMPLEX_DTYPE
import numpy as np
from typing import List
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import tensor_shape

# ========================================================
#  Abstract Gate class
# ========================================================

class Gate(Layer):
    """
    Abstract Quantum Gate class

    """

    def __init__(self, nparams: int, wires: List[int], value: List[float] = None, name=None, **kwargs):
        """
        The ``Gate`` class performs unitary quantum gates on the tensor subspace of a wave function

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

        super(Gate, self).__init__(name=name, **kwargs)
        self.nparams = nparams
        self.value = value
        self.nqubits = len(wires)
        self.external_input = None
        # This boolean indicates whether this layer is the first layer is the first layer.
        self._input_layer = False
        self._batch_params = False
        self._batch_size = None
        self.layer = None
        # shift all dimensions due to batch dim
        self.wires = [w + 1 for w in wires]

    def set_external_input(self, external_input: tf.Tensor):
        """
        Connect an external tensor to the gate parameters. Tensor must have shape (nparams, 1).
        This function enables the construction of hybrid architectures by having the parameters of the circuit be fed
        from some external model.

        Args:
            *external_input (Tensor)*:
                Tensor of shape (nparams, 1) with gate parameters.

        Returns (inplace):
            None

        """
        assert self.nparams > 0, "This gate has no parameters that can be controlled externally"
        assert tf.is_tensor(external_input), "external input must be a tf.Tensor, received {}".format(
            type(external_input))
        assert tf.compat.v1.debugging.assert_type(external_input, tf_type=TF_FLOAT_DTYPE,
                                                  message="External input Tensor must have type float32, ")
        self.external_input = external_input

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
        super(Gate, self).build(input_shape)
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        assert len(input_shape)>1, "Input shape must have at least dim>1, received {}".format(input_shape)
        # Check that only the first dimension is can be the batch dimension.
        if None in input_shape:
            assert ((input_shape[0] == None)) & (
                    input_shape.count(None) == 1), "Only the first dimension can be None, found {}".format(
                input_shape)
            if self._input_layer:
                assert (self._batch_params == False), \
                    "We cannot have a batch of wave functions and a batch of parameters at the same time. " \
                    "Received {} for the first layer and self.batch_params={}".format(
                        input_shape, self._batch_params)
        elif (input_shape[0] > 1):
            assert all([input_shape[i] == 2 for i in range(1, len(input_shape))]), \
                "Input vector needs to have shape (x, 2, 2,...,2), found {}".format(input_shape)
            if self._input_layer:
                assert (self._batch_params == False), \
                    "We cannot have a batch of wave functions and a batch of parameters at the same time. " \
                    "Received {} for the first layer and self.batch_params={}".format(
                        input_shape, self._batch_params)
        else:
            assert (input_shape[0] == 1) & (all([input_shape[i] == 2 for i in range(1, len(input_shape))])), \
                "Input vector needs to have shape (1, 2, 2,...,2), found {}".format(input_shape)

        # Assert that the wires are set correctly.
        self.total_qubits = len(input_shape) - 1
        assert max(self.wires) - 1 < self.total_qubits, \
            "Operation {} is performed on qubit {}, but system only has {} qubits".format(
                self.name, max(self.wires) - 1, self.total_qubits)
        # Set the initial parameters of the gate.
        if self.nparams == 0:
            assert self.value is None, "Initial value given for gate without parameters"
            self.theta = None
        else:
            with tf.compat.v1.variable_scope(name_or_scope=None, default_name="gate_init"):
                if self._batch_params:
                    assert tf.is_tensor(self.external_input), "batch_params=True, but no external input provided."
                # Set the gate parameters to the external input, if necessary.
                if tf.is_tensor(self.external_input):
                    ext_input_shape = self.external_input.shape.as_list()
                    if ((ext_input_shape[0] == None) or (ext_input_shape[0] > 1)):
                        assert self._batch_params, "Received parameters with shape {}, but batch_params = {}, pass the batch_params=True argument " \
                                                   "to the QuantumCircuit class to enable batches of parameters.".format(
                            self.external_input.shape.as_list(), self._batch_params)
                        assert ext_input_shape[1:] == [self.nparams, 1], \
                            "external input must have shape {}, received {}".format(
                                [None, self.nparams, 1], self.external_input.shape.as_list()
                            )
                    else:
                        assert ext_input_shape == [1, self.nparams, 1], \
                            "external input must have shape {}, received {}".format(
                                [1, self.nparams, 1], self.external_input.shape.as_list()
                            )
                    self.theta = self.external_input
                elif not self.trainable:
                    if self.value is not None:
                        self.theta = tf.convert_to_tensor(self.value)
                    else:
                        self.theta = tf.random.uniform(1) * 2 * np.pi
                else:
                    # If we need to set an initial value, we do it here, otherwise choose uniform start value
                    if self.value is None:
                        self.theta = tf.compat.v1.get_variable(
                            initializer=tf.constant(np.random.uniform(size=(self.nparams, 1)), dtype=TF_FLOAT_DTYPE),
                            # shape=(self.nparams, 1),
                            dtype=TF_FLOAT_DTYPE,
                            name=self.name,
                            trainable=self.trainable)
                    else:
                        if tf.is_tensor(self.value):
                            self.theta = tf.reshape(self.value,(-1,1))
                        else:
                            self.value = np.array(self.value, dtype=np.float32).reshape((-1, 1))
                            assert len(
                                self.value) == self.nparams, "Gate requires {} initial values, but {} were given".format(
                                self.nparams, len(self.value))
                            self.theta = tf.compat.v1.get_variable(initializer=self.value, dtype=TF_FLOAT_DTYPE,
                                                                   name=self.name,
                                                                   trainable=self.trainable)
        # Get the inverse_permutation for the tensordot
        unused_idxs = [idx for idx in range(1, self.total_qubits + 1) if idx not in self.wires]
        perm = self.wires + [0] + unused_idxs
        self.inv_perm = list(np.argsort(perm))
        self.tdot_axes = [list(range(self.nqubits, 2 * self.nqubits)), self.wires]
        self.built = True

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic goes here.

        """

        raise NotImplementedError


# ============
#  Fixed gates
# ============

def identity():
    return tf.convert_to_tensor(np.eye(2), dtype=TF_COMPLEX_DTYPE)


def pauli_x():
    return tf.convert_to_tensor(np.array([[0, 1], [1, 0]]), dtype=TF_COMPLEX_DTYPE)


def pauli_y():
    return tf.convert_to_tensor(np.array([[0, -1j], [1j, 0]]), dtype=TF_COMPLEX_DTYPE)


def pauli_z():
    return tf.convert_to_tensor(np.array([[1, 0], [0, -1]]), dtype=TF_COMPLEX_DTYPE)


def hadamard():
    return tf.convert_to_tensor(np.array([[1, 1], [1, -1]]) / np.math.sqrt(2), dtype=TF_COMPLEX_DTYPE)


def swap():
    return tf.convert_to_tensor(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), dtype=TF_COMPLEX_DTYPE)


def cnot():
    return tf.convert_to_tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]), dtype=TF_COMPLEX_DTYPE)


def cz():
    return tf.convert_to_tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
                                dtype=TF_COMPLEX_DTYPE)


def toffoli():
    arr = np.eye(8)
    arr[6, 6] = 0
    arr[6, 7] = 1
    arr[7, 7] = 0
    arr[7, 6] = 1
    return tf.convert_to_tensor(arr, dtype=TF_COMPLEX_DTYPE)


class Hadamard(Gate):
    r"""
    Gate that implements the Hadamard unitary operation.

    .. math::

        \text{H} = \frac{1}{\sqrt{2}}
        \begin{pmatrix} 
        1 & 1 \\
        1& -1
        \end{pmatrix}

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'H_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on one qubit, found {}".format(wires)
        super(Hadamard, self).__init__(nparams=0, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(Hadamard, self).build(input_shape)

        self.op = hadamard()
        if self.conjugate:
            self.op = tf.transpose(self.op, conjugate=True)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        phi = tensordot(self.op, inputs, axes=self.tdot_axes)
        return tf.transpose(phi, perm=self.inv_perm)


class PauliX(Gate):
    r"""
    Gate that implements the Pauli X unitary operation.

    .. math::

        \sigma^x =
        \begin{pmatrix}
        0 & 1  \\
        1 & 0
        \end{pmatrix}

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'X_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on one qubit, found {}".format(wires)
        super(PauliX, self).__init__(nparams=0, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(PauliX, self).build(input_shape)

        self.op = pauli_x()
        if self.conjugate:
            self.op = tf.transpose(self.op, conjugate=True)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        phi = tensordot(self.op, inputs, axes=self.tdot_axes)
        return tf.transpose(phi, perm=self.inv_perm)


class PauliY(Gate):
    r"""
    Gate that implements the Pauli Y unitary operation.

    .. math::

        \sigma^y =
        \begin{pmatrix}
        0 & -i  \\
        i & 0
        \end{pmatrix}

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'Y_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on one qubit, found {}".format(wires)
        super(PauliY, self).__init__(nparams=0, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(PauliY, self).build(input_shape)

        self.op = pauli_y()
        if self.conjugate:
            self.op = tf.transpose(self.op, conjugate=True)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        phi = tensordot(self.op, inputs, axes=self.tdot_axes)
        return tf.transpose(phi, perm=self.inv_perm)


class PauliZ(Gate):
    r"""
    Gate that implements the Pauli Z unitary operation.

    .. math::

        \sigma^z =
        \begin{pmatrix}
        1 & 0 \\
        0 & -1
        \end{pmatrix}

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'Z_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on one qubit, found {}".format(wires)
        super(PauliZ, self).__init__(nparams=0, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(PauliZ, self).build(input_shape)

        self.op = pauli_z()
        if self.conjugate:
            self.op = tf.transpose(self.op, conjugate=True)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """

        phi = tensordot(self.op, inputs, axes=self.tdot_axes)
        return tf.transpose(phi, perm=self.inv_perm)


class Swap(Gate):
    r"""
    Gate that implements the SWAP unitary operation.

    .. math::

        \sqrt{\text{SWAP}} =
        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 1 \\
        \end{pmatrix}

        """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'SWAP_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 2, "Gate should operate on two qubits, found {}".format(wires)
        super(Swap, self).__init__(nparams=0, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(Swap, self).build(input_shape)

        self.op = swap()
        if self.conjugate:
            self.op = tf.transpose(self.op, conjugate=True)

        self.op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        phi = tensordot(self.op, inputs, axes=self.tdot_axes)
        return tf.transpose(phi, perm=self.inv_perm)


class CNOT(Gate):
    r"""
    Gate that implements the CNOT unitary operation.

    .. math::

        \text{CNOT} =
        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 1 \\
        0 & 0 & 1 & 0 \\
        \end{pmatrix}

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'CNOT_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 2, "Gate should operate on two qubits, found {}".format(wires)
        super(CNOT, self).__init__(nparams=0, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(CNOT, self).build(input_shape)

        self.op = cnot()
        if self.conjugate:
            self.op = tf.transpose(self.op, conjugate=True)

        self.op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        phi = tensordot(self.op, inputs, axes=self.tdot_axes)
        return tf.transpose(phi, perm=self.inv_perm)


class CZ(Gate):
    r"""
    Gate that implements the CZ unitary operation.

    .. math::

        \text{CZ} =
        \begin{pmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 \\
        0 & 0 & 1 & 0 \\
        0 & 0 & 0 & -1 \\
        \end{pmatrix}

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'CZ_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 2, "Gate should operate on two qubits, found {}".format(wires)
        super(CZ, self).__init__(nparams=0, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(CZ, self).build(input_shape)

        self.op = cz()
        if self.conjugate:
            self.op = tf.transpose(self.op, conjugate=True)
        self.op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        phi = tensordot(self.op, inputs, axes=self.tdot_axes)
        return tf.transpose(phi, perm=self.inv_perm)


# ==================
# Parametrized gates
# ==================


class Phase(Gate):
    r"""
    Gate that implements a phase rotation

    .. math::

        R(\theta) =
        \begin{pmatrix}
        1 & 0\\
        0 & e^{i\theta}
        \end{pmatrix}

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'Phase_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on one qubit, found {}".format(wires)
        super(Phase, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(Phase, self).build(input_shape)

        self.n_0 = tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE)
        self.n_1 = tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = self.n_0 + tf.math.exp(tf.complex(0., self.theta)) * self.n_1

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = self.n_0 + tf.math.exp(tf.complex(0., self.theta)) * self.n_1

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class RX(Gate):
    r"""
    Gate that implements a rotation around the spin x-axis.

    .. math::

        R_X(\theta) = \exp(-i\theta\sigma^x/2) =
        \cos(\theta/2) \: I + i \sin(-\theta/2) \sigma^x

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'RX_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on one qubit, found {}".format(wires)
        super(RX, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(RX, self).build(input_shape)

        self.I = identity()
        self.PX = pauli_x()

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic

        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] == 1:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PX

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            self.tdot_axes[1] = [w - 1 for w in self.tdot_axes[1]]
            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[0] / 2)) * self.PX

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class RY(Gate):
    r"""
    Gate that implements a rotation around the spin y-axis.

    .. math::

        R_Y(\theta) = \exp(-i\theta\sigma^y/2) =
        \cos(\theta/2) \: I + i \sin(-\theta/2) \sigma^y

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'RY_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on one qubit, found {}".format(wires)
        super(RY, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(RY, self).build(input_shape)

        self.I = identity()
        self.PY = pauli_y()

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] == 1:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PY

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            self.tdot_axes[1] = [w - 1 for w in self.tdot_axes[1]]
            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.cast(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[0] / 2)) * self.PY

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class RZ(Gate):
    r"""
    Gate that implements a rotation around the spin z-axis.

    .. math::

        R_Z(\theta) = \exp(-i\theta\sigma^z/2) =
        \cos(\theta/2) \: I + i \sin(-\theta/2) \sigma^z

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'RZ_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on one qubit, found {}".format(wires)
        super(RZ, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(RZ, self).build(input_shape)

        self.I = identity()
        self.PZ = pauli_z()

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] == 1:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PZ

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            self.tdot_axes[1] = [w - 1 for w in self.tdot_axes[1]]
            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)

            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[0] / 2)) * self.PZ

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class R3(Gate):
    r"""
    Gate that implements a rotation around an arbitrary spin axis.

    .. math::

        R_3(\theta, \phi, \gamma) = R_Z(\theta) R_Y(\phi) R_Z(\gamma)

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'R3_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on one qubit, found {}".format(wires)
        super(R3, self).__init__(nparams=3, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(R3, self).build(input_shape)

        self.I = identity()
        self.PY = pauli_y()
        self.PZ = pauli_z()

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            rz1 = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PZ
            ry = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PY
            rz2 = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PZ
            op = rz1 @ (ry @ rz2)

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            rz1 = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[0] / 2)) * self.PZ
            ry = tf.complex(tf.math.cos(self.theta[1] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[1] / 2)) * self.PY
            rz2 = tf.complex(tf.math.cos(self.theta[2] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[2] / 2)) * self.PZ
            op = rz1 @ (ry @ rz2)

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class CRX(Gate):
    r"""
    Gate that implements a conditional rotation around the spin x-axis.

    .. math::

        \text{C}R_X(\theta) =
        |0\rangle \langle 0| \otimes I + |1\rangle \langle 1| \otimes R_x(\theta)

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'CRX_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 2, "Gate should operate on two qubits, found {}".format(wires)
        super(CRX, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(CRX, self).build(input_shape)

        self.I = identity()
        self.PX = pauli_x()

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            rx = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PX
            op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
                tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), rx)

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            rx = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[0] / 2)) * self.PX
            op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
                tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), rx)

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class CRY(Gate):
    r"""
    Gate that implements a conditional rotation around the spin y-axis.

    .. math::

        \text{C}R_Y(\theta) =
        |0\rangle \langle 0| \otimes I + |1\rangle \langle 1| \otimes R_y(\theta)

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'CRY_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 2, "Gate should operate on two qubits, found {}".format(wires)
        super(CRY, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(CRY, self).build(input_shape)

        self.I = identity()
        self.PY = pauli_y()

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            ry = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PY
            op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
                tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), ry)

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            ry = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[0] / 2)) * self.PY
            op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
                tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), ry)

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class CRZ(Gate):
    r"""
    Gate that implements a conditional rotation around the spin z-axis.

    .. math::

        \text{C}R_Z(\theta) =
        |0\rangle \langle 0| \otimes I + |1\rangle \langle 1| \otimes R_z(\theta)

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'CRZ_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 2, "Gate should operate on two qubits, found {}".format(wires)
        super(CRZ, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(CRZ, self).build(input_shape)

        self.I = identity()
        self.PZ = pauli_z()

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            rz = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PZ
            op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
                tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), rz)

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            rz = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[0] / 2)) * self.PZ
            op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
                tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), rz)

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class CR3(Gate):
    r"""
    Gate that implements a conditional rotation around an arbitrary spin axis.

    .. math::

        \text{C}R_3(\theta,\phi, \gamma) =
        |0\rangle \langle 0| \otimes I + |1\rangle \langle 1| \otimes R_3(\theta,\phi, \gamma)

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'CR3_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 2, "Gate should operate on two qubits, found {}".format(wires)
        super(CR3, self).__init__(nparams=3, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(CR3, self).build(input_shape)

        self.I = identity()
        self.PY = pauli_y()
        self.PZ = pauli_z()

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            rz1 = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[0] / 2)) * self.PZ
            ry = tf.complex(tf.math.cos(self.theta[1] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[1] / 2)) * self.PY
            rz2 = tf.complex(tf.math.cos(self.theta[2] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[2] / 2)) * self.PZ
            r3 = rz1 @ (ry @ rz2)
            op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
                tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), r3)

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            rz1 = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[0] / 2)) * self.PZ
            ry = tf.complex(tf.math.cos(self.theta[1] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[1] / 2)) * self.PY
            rz2 = tf.complex(tf.math.cos(self.theta[2] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta[2] / 2)) * self.PZ
            r3 = rz1 @ (ry @ rz2)
            op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
                tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), r3)

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)
class XX(Gate):
    r"""
    Gate that implements a conditional rotation Ising XX spin interaction.

    .. math::

        \text{XX}(\theta) =  \exp(-i\theta\sigma^x \otimes \sigma_x/2) = \cos(\theta/2) \: I + i \sin(-\theta/2) \sigma^x \otimes \sigma_x

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'XX_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 2, "Gate should operate on two qubits, found {}".format(wires)
        super(XX, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(XX, self).build(input_shape)

        self.I = tf_kron(identity(), identity())
        self.PX = tf_kron(pauli_x(), pauli_x())

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PX

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PX

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class YY(Gate):
    r"""
    Gate that implements a conditional rotation Ising YY spin interaction.

    .. math::

        \text{YY}(\theta) =  \exp(-i\theta\sigma^y \otimes \sigma_y/2) = \cos(\theta/2) \: I + i \sin(-\theta/2) \sigma^y \otimes \sigma_y

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'YY_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 2, "Gate should operate on two qubits, found {}".format(wires)
        super(YY, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(YY, self).build(input_shape)

        self.I = tf_kron(identity(), identity())
        self.PY = tf_kron(pauli_y(), pauli_y())

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PY

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PY

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class ZZ(Gate):
    r"""
    Gate that implements a conditional rotation Ising ZZ spin interaction.

    .. math::

        \text{ZZ}(\theta) =  \exp(-i\theta\sigma^z \otimes \sigma_z/2) = \cos(\theta/2) \: I + i \sin(-\theta/2) \sigma^z \otimes \sigma_z

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'ZZ_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 2, "Gate should operate on two qubits, found {}".format(wires)
        super(ZZ, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(ZZ, self).build(input_shape)

        self.I = tf_kron(identity(), identity())
        self.PZ = tf_kron(pauli_z(), pauli_z())

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PZ

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                -self.theta / 2)) * self.PZ

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)

class Toffoli(Gate):
    r"""
    Gate that implements a controlled-controlled flip by applying the Toffoli gate.

    .. math::

        \text{CCNOT} =
        \begin{pmatrix}
        1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
        0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
        0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
        \end{pmatrix}

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'CR3_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 3, "Gate should operate on three qubits, found {}".format(wires)
        super(Toffoli, self).__init__(nparams=3, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(Toffoli, self).build(input_shape)

        self.op = toffoli()
        if self.conjugate:
            self.op = tf.transpose(self.op, conjugate=True)
        self.op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        phi = tensordot(self.op, inputs, axes=self.tdot_axes)
        return tf.transpose(phi, perm=self.inv_perm)


def hermitian(operation: tf.Tensor) -> tf.Tensor:
    r"""
    Validate that operation is hermitian

    Args:
        *operation (array)*: Square hermitian matrix.

    Returns (Tensor):
         Square hermitian matrix.

    """
    # TODO: more checks?

    # if operation.shape[0] != operation.shape[1]:
    #     raise ValueError("Expectation must be a square matrix.")
    # tf.debugging.assert_shapes()
    tf.compat.v1.debugging.assert_type(operation, TF_COMPLEX_DTYPE,
                                       message="Expected operation of type {}, got {}".format(TF_COMPLEX_DTYPE,
                                           operation.dtype.name))
    # tf.debugging.assert_near(x=operation, y=tf.transpose(operation, conjugate=True), rtol=tf.cast(1e-8, TF_FLOAT_DTYPE),
    # message="Expectation must be Hermitian.")

    return operation
