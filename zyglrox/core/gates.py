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
from zyglrox.core.utils import tf_kron, tensordot
from zyglrox.core._config import TF_FLOAT_DTYPE, TF_COMPLEX_DTYPE
import numpy as np
from typing import List, Union
from tensorflow.keras.layers import Layer
from tensorflow.python.framework import tensor_shape
import string
import copy


# ========================================================
#  Abstract Gate class
# ========================================================
class Gate(Layer):
    """
    Abstract Quantum Gate class

    """
    ALPHABET = list(string.ascii_lowercase)

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
        # self.external_input = None
        # This boolean indicates whether this layer is the first layer is the first layer.
        self._input_layer = False
        self._batch_params = False
        self._batch_size = None
        self.layer = None
        self._is_projector = False
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
        self.value = external_input

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
        assert len(input_shape) > 1, "Input shape must have at least dim>1, received {}".format(input_shape)
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
                    assert tf.is_tensor(self.value), "batch_params=True, but no external input provided."
                if not self.trainable:
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
                            self.value = tf.reshape(self.value, (-1, self.nparams, 1))
                            value_input_shape = self.value.shape.as_list()
                            if ((value_input_shape[0] == None) or (value_input_shape[0] > 1)):
                                assert self._batch_params, "Received parameters with shape {}, but batch_params = {}, pass the batch_params=True argument " \
                                                           "to the QuantumCircuit class to enable batches of parameters.".format(
                                    value_input_shape, self._batch_params)
                                assert value_input_shape[1:] == [self.nparams, 1], \
                                    "external input must have shape {}, received {}".format(
                                        [None, self.nparams, 1], value_input_shape
                                    )
                            else:
                                assert value_input_shape == [1, self.nparams, 1], \
                                    "external input must have shape {}, received {}".format(
                                        [1, self.nparams, 1], value_input_shape
                                    )
                            self.theta = self.value
                        else:
                            self.value = np.array(self.value, dtype=float).reshape((-1, self.nparams, 1))
                            assert self.value.size == self.nparams, "Gate requires {} initial values, but {} were given".format(
                                self.nparams, self.value.size)
                            self.theta = tf.compat.v1.get_variable(initializer=self.value, dtype=TF_FLOAT_DTYPE,
                                                                   name=self.name,
                                                                   trainable=self.trainable)
        if self._batch_params & ~self._input_layer:
            self.wires = [w - 1 for w in self.wires]
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

    def get_full_unitary(self, N):
        """
        Takes the unitary and promotes it to an operator on the full unitary space C^{2^n x 2^n}.

        Returns:
            Numpy array of size 2^n x 2^n corresponding to the full unitary.
        """
        assert self.built, '`Gate` must be build first, call execute() on the QuantumCircuit'
        U = tf.reshape(tf.eye(2**N, dtype=TF_COMPLEX_DTYPE),[2]*2*N)
        op = tf.reshape(self.op, [2] * 2 * len(self.wires))
        einsum_indices_operator = list(range(2 * len(self.wires)))
        einsum_indices_final_operator = list(range(2 * len(self.wires), 2 * N + 2 * len(self.wires)))

        for i, w in enumerate(self.wires):
            einsum_indices_final_operator[w-1] = einsum_indices_operator[i + len(self.wires)]
        einsum_indices_operator_out = copy.copy(einsum_indices_final_operator)
        for i, w in enumerate(self.wires):
            einsum_indices_operator_out[w-1] = einsum_indices_operator[i]

        einsum_indices_operator = [self.ALPHABET[s] for s in einsum_indices_operator]
        einsum_indices_final_operator = [self.ALPHABET[s] for s in einsum_indices_final_operator]
        einsum_indices_operator_out = [self.ALPHABET[s] for s in einsum_indices_operator_out]
        einsum_contraction = ''.join(einsum_indices_operator) + ',' + ''.join(einsum_indices_final_operator) + '->' + ''.join(
            einsum_indices_operator_out)
        final_operator = tf.linalg.einsum(einsum_contraction, op, U)
        return tf.reshape(final_operator, (2 ** N, 2 ** N))


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


ps_to_tensor = {'X': pauli_x(), 'Y': pauli_y(), 'Z': pauli_z()}


def hadamard():
    return tf.convert_to_tensor(np.array([[1, 1], [1, -1]]) / np.math.sqrt(2), dtype=TF_COMPLEX_DTYPE)


def t_gate():
    return tf.convert_to_tensor(np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]]), dtype=TF_COMPLEX_DTYPE)


def swap():
    return tf.convert_to_tensor(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
                                dtype=TF_COMPLEX_DTYPE)


def cnot():
    return tf.convert_to_tensor(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]),
                                dtype=TF_COMPLEX_DTYPE)


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


class T(Gate):
    r"""
    Gate that implements the T unitary operation.

    .. math::

        \text{H} = \frac{1}{\sqrt{2}}
        \begin{pmatrix}
        1 & 0 \\
        0& e^{i \pi/4}
        \end{pmatrix}

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'T_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on one qubit, found {}".format(wires)
        super(T, self).__init__(nparams=0, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(T, self).build(input_shape)

        self.op = t_gate()
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
    Gate that implements the CNOT unitary operation. Order of wires is c, t

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
        if (self._batch_params):

            self.op = self.n_0 + tf.math.exp(tf.complex(0., self.theta)) * self.n_1
        else:
            self.op = self.n_0 + tf.math.exp(tf.complex(0., self.theta[0])) * self.n_1

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(self.op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)

            phi = tensordot(self.op, inputs, axes=self.tdot_axes)
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
        if (self._batch_params):
            self.op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PX
        else:
            self.op = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[0] / 2)) * self.PX

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic

        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] == 1:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            self.tdot_axes[1] = [w - 1 for w in self.tdot_axes[1]]
            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(self.op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)

            phi = tensordot(self.op, inputs, axes=self.tdot_axes)
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
        if (self._batch_params):
            self.op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PY
        else:
            self.op = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[0] / 2)) * self.PY

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] == 1:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            self.tdot_axes[1] = [w - 1 for w in self.tdot_axes[1]]
            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(self.op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)

            phi = tensordot(self.op, inputs, axes=self.tdot_axes)
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
        if (self._batch_params):

            self.op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PZ
        else:
            self.op = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[0] / 2)) * self.PZ

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] == 1:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            self.tdot_axes[1] = [w - 1 for w in self.tdot_axes[1]]
            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(self.op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)

            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)

            phi = tensordot(self.op, inputs, axes=self.tdot_axes)
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
        if (self._batch_params):
            rz1 = tf.complex(tf.math.cos(self.theta[:, 0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[:, 0] / 2)) * self.PZ
            ry = tf.complex(tf.math.cos(self.theta[:, 1] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[:, 1] / 2)) * self.PY
            rz2 = tf.complex(tf.math.cos(self.theta[:, 2] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[:, 2] / 2)) * self.PZ
        else:
            rz1 = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[0] / 2)) * self.PZ
            ry = tf.complex(tf.math.cos(self.theta[1] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[1] / 2)) * self.PY
            rz2 = tf.complex(tf.math.cos(self.theta[2] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[2] / 2)) * self.PZ
        self.op = rz1 @ (ry @ rz2)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(self.op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)

            phi = tensordot(self.op, inputs, axes=self.tdot_axes)
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
        if (self._batch_params):

            rx = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PX
        else:
            rx = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[0] / 2)) * self.PX
        self.op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
            tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), rx)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

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
        if (self._batch_params):

            ry = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PY
        else:
            ry = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[0] / 2)) * self.PY
        self.op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
            tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), ry)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

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
        if (self._batch_params):

            rz = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PZ
        else:
            rz = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[0] / 2)) * self.PZ
        self.op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
            tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), rz)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

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
        if (self._batch_params):

            rz1 = tf.complex(tf.math.cos(self.theta[:, 0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[:, 0] / 2)) * self.PZ
            ry = tf.complex(tf.math.cos(self.theta[:, 1] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[:, 1] / 2)) * self.PY
            rz2 = tf.complex(tf.math.cos(self.theta[:, 2] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[:, 2] / 2)) * self.PZ
            r3 = rz1 @ (ry @ rz2)
        else:
            rz1 = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[0] / 2)) * self.PZ
            ry = tf.complex(tf.math.cos(self.theta[1] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[1] / 2)) * self.PY
            rz2 = tf.complex(tf.math.cos(self.theta[2] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[2] / 2)) * self.PZ
            r3 = rz1 @ (ry @ rz2)
        self.op = tf_kron(tf.convert_to_tensor(np.array([[1, 0], [0, 0]]), dtype=TF_COMPLEX_DTYPE), self.I) + tf_kron(
            tf.convert_to_tensor(np.array([[0, 0], [0, 1]]), dtype=TF_COMPLEX_DTYPE), r3)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

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
        if (self._batch_params):

            self.op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PX
        else:
            self.op = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[0] / 2)) * self.PX

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

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
        if (self._batch_params):

            self.op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PY
        else:
            self.op = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta[0] / 2)) * self.PY

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

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
        if (self._batch_params):

            self.op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(-self.theta / 2)) * self.PZ
        else:
            self.op = tf.complex(tf.math.cos(self.theta[0] / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(-self.theta[0] / 2)) * self.PZ

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs, multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[-1] + [2] * self.nqubits * 2)
            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)

            return phi

        else:

            if self.conjugate:
                self.op = tf.transpose(self.op, conjugate=True)
            op = tf.reshape(self.op, shape=[2] * self.nqubits * 2)

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


class XX_hexa(Gate):
    r"""
    Gate that implements a 6 qubit rotation Ising XX spin interaction.

    .. math::

        \text{XX}(\theta) =  \exp(-i\theta \bigotimes^6\sigma^x/2) = \cos(\theta/2) \: I + i \sin(-\theta/2) \bigotimes^6 \sigma^x

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'XXhexa_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 6, "Gate should operate on 6 qubits, found {}".format(wires)
        super(XX_hexa, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(XX_hexa, self).build(input_shape)

        self.I = tf_kron(tf_kron(identity(), identity()),
                         tf_kron(tf_kron(identity(), identity()), tf_kron(identity(), identity())))
        self.PX = tf_kron(tf_kron(pauli_x(), pauli_x()),
                          tf_kron(tf_kron(pauli_x(), pauli_x()), tf_kron(pauli_x(), pauli_x())))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs,
                                 multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PX

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PX

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class YY_hexa(Gate):
    r"""
    Gate that implements a 6 qubit rotation Ising YY spin interaction.

    .. math::

        \text{XX}(\theta) =  \exp(-i\theta \bigotimes^6\sigma^x/2) = \cos(\theta/2) \: I + i \sin(-\theta/2) \bigotimes^6 \sigma^y

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'YYhexa_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 6, "Gate should operate on 6 qubits, found {}".format(wires)
        super(YY_hexa, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(YY_hexa, self).build(input_shape)

        self.I = tf_kron(tf_kron(identity(), identity()),
                         tf_kron(tf_kron(identity(), identity()), tf_kron(identity(), identity())))
        self.PY = tf_kron(tf_kron(pauli_y(), pauli_y()),
                          tf_kron(tf_kron(pauli_y(), pauli_y()), tf_kron(pauli_y(), pauli_y())))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs,
                                 multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PY

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PY

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class ZZ_hexa(Gate):
    r"""
    Gate that implements a 6 qubit rotation Ising XX spin interaction.

    .. math::

        \text{ZZ}(\theta) =  \exp(-i\theta \bigotimes^6\sigma^z/2) = \cos(\theta/2) \: I + i \sin(-\theta/2) \bigotimes^6 \sigma^z

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'ZZhexa_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 6, "Gate should operate on 6 qubits, found {}".format(wires)
        super(ZZ_hexa, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(ZZ_hexa, self).build(input_shape)

        self.I = tf_kron(tf_kron(identity(), identity()),
                         tf_kron(tf_kron(identity(), identity()), tf_kron(identity(), identity())))
        self.PZ = tf_kron(tf_kron(pauli_z(), pauli_z()),
                          tf_kron(tf_kron(pauli_z(), pauli_z()), tf_kron(pauli_z(), pauli_z())))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs,
                                 multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PZ

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PZ

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class XX_tri(Gate):
    r"""
    Gate that implements a 6 qubit rotation Ising XX spin interaction.

    .. math::

        \text{ZZ}(\theta) =  \exp(-i\theta \bigotimes^6\sigma^z/2) = \cos(\theta/2) \: I + i \sin(-\theta/2) \bigotimes^6 \sigma^z

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'XXtri_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 3, "Gate should operate on 3 qubits, found {}".format(wires)
        super(XX_tri, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(XX_tri, self).build(input_shape)

        self.I = tf_kron(tf_kron(identity(), identity()), identity())
        self.PX = tf_kron(tf_kron(pauli_x(), pauli_x()), pauli_x())

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs,
                                 multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PX

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PX

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class YY_tri(Gate):
    r"""
    Gate that implements a 6 qubit rotation Ising XX spin interaction.

    .. math::

        \text{ZZ}(\theta) =  \exp(-i\theta \bigotimes^6\sigma^z/2) = \cos(\theta/2) \: I + i \sin(-\theta/2) \bigotimes^6 \sigma^z

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'YYtri_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 3, "Gate should operate on 3 qubits, found {}".format(wires)
        super(YY_tri, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(YY_tri, self).build(input_shape)

        self.I = tf_kron(tf_kron(identity(), identity()), identity())
        self.PY = tf_kron(tf_kron(pauli_y(), pauli_y()), pauli_y())

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs,
                                 multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PY

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PY

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class ZZ_tri(Gate):
    r"""
    Gate that implements a 6 qubit rotation Ising XX spin interaction.

    .. math::

        \text{ZZ}(\theta) =  \exp(-i\theta \bigotimes^6\sigma^z/2) = \cos(\theta/2) \: I + i \sin(-\theta/2) \bigotimes^6 \sigma^z

    """

    def __init__(self, wires: List[int], value: List[float] = None, conjugate=False, name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'ZZtri_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 3, "Gate should operate on 3 qubits, found {}".format(wires)
        super(ZZ_tri, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate

    def build(self, input_shape):
        super(ZZ_tri, self).build(input_shape)

        self.I = tf_kron(tf_kron(identity(), identity()), identity())
        self.PZ = tf_kron(tf_kron(pauli_z(), pauli_z()), pauli_z())

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs,
                                 multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PZ

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.PZ

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


def kron_recursion(ops, kron_op=None, i=0):
    if i == (len(ops) - 1):
        return kron_op
    elif not i:
        kron_op = tf_kron(ops[i], ops[i + 1])
        return kron_recursion(ops, kron_op, i + 1)
    else:
        kron_op = tf_kron(kron_op, ops[i + 1])
        return kron_recursion(ops, kron_op, i + 1)


class PauliRotation(Gate):
    r"""
    Gate that implements a 6 qubit rotation Ising XX spin interaction.

    .. math::

        \text{ZZ}(\theta) =  \exp(-i\theta \bigotimes^6\sigma^z/2) = \cos(\theta/2) \: I + i \sin(-\theta/2) \bigotimes^6 \sigma^z

    """

    def __init__(self, paulistring: str, wires: List[int], value: List[float] = None, conjugate=False, name=None,
                 **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert isinstance(paulistring, str), "'paulistring' must be a string"
        assert all([ps in ['X', 'Y', 'Z'] for ps in
                    paulistring]), f"'paulistring' must consist of X,Y,Z strings, received {paulistring}"
        assert len(paulistring) == len(wires), "'wires' must have the same length as 'paulistring'"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'PauliRot' + '_'.join([str(s) for s in wires])
        super(PauliRotation, self).__init__(nparams=1, wires=wires, value=value, name=name, **kwargs)
        self.conjugate = conjugate
        self.paulistring = [ps.upper() for ps in paulistring]

    def build(self, input_shape):
        super(PauliRotation, self).build(input_shape)

        self.Prot = kron_recursion([ps_to_tensor[ps] for ps in self.paulistring])
        I = identity()
        self.I = kron_recursion([I] * len(self.paulistring))

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            # if we have batches of parameters and inputs does not already have a batch dim, we tile to give it one
            if inputs.shape.as_list()[0] is not None:
                inputs = tf.tile(inputs,
                                 multiples=[tf.shape(self.theta)[0], *[1 for _ in range(self.total_qubits)]])

            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.Prot

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[-1] + [2] * self.nqubits * 2)

            phi = tf.map_fn(lambda x: tensordot(x[0], x[1], axes=self.tdot_axes), elems=(op, inputs),
                            dtype=(TF_COMPLEX_DTYPE), parallel_iterations=self._batch_size)
            return phi

        else:
            op = tf.complex(tf.math.cos(self.theta / 2), tf.constant(0., TF_FLOAT_DTYPE)) * self.I + tf.complex(
                tf.constant(0., TF_FLOAT_DTYPE), tf.math.sin(
                    -self.theta / 2)) * self.Prot

            if self.conjugate:
                op = tf.transpose(op, conjugate=True)
            op = tf.reshape(op, shape=[2] * self.nqubits * 2)

            phi = tensordot(op, inputs, axes=self.tdot_axes)
            return tf.transpose(phi, perm=self.inv_perm)


class Projector(Gate):
    r"""
    Projector that projects the state onto an eigenvector of a Hermitian observable

    .. math::

        \rho^\prime = \frac{\Pi_n \rho \Pi_n}{Z}

    """

    def __init__(self, pi: Union[np.ndarray, tf.Tensor], wires: List[int], name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'Projector_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on 1 qubits, found {}".format(wires)
        assert isinstance(pi,
                          (np.ndarray, tf.Tensor)), f"'pi' must be of type np.ndarray or tf.Tensor, received {type(pi)}"

        assert pi.shape == (2, 2), f"pi must have shape [2,2], received {pi.shape}"
        if isinstance(pi, np.ndarray):
            self.op = tf.constant(pi, dtype=TF_COMPLEX_DTYPE)
        else:
            self.op = pi
        super(Projector, self).__init__(nparams=0, wires=wires, value=None, name=name, **kwargs)
        self.conjugate = False
        self._is_projector = True

    def build(self, input_shape):
        super(Projector, self).build(input_shape)

        einsum_indices_pi = self.ALPHABET[0] + self.ALPHABET[1]
        einsum_indices_phi = [self.ALPHABET[-1]] + [self.ALPHABET[i + 2] for i in range(len(input_shape) - 1)]
        einsum_indices_phi[self.wires[0]] = self.ALPHABET[1]
        einsum_indices_phi_out = copy.copy(einsum_indices_phi)
        einsum_indices_phi_out[self.wires[0]] = self.ALPHABET[0]
        self.einsum_contraction = ''.join(einsum_indices_pi) + ',' + ''.join(einsum_indices_phi) + '->' + ''.join(
            einsum_indices_phi_out)

    def call(self, inputs: tf.Tensor, probability_only: bool = False, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            raise NotImplementedError

        else:
            phi_out = tf.einsum(self.einsum_contraction, self.op, inputs)
            self.Z = tf.linalg.norm(tf.reshape(phi_out, (-1, 1)))
            if probability_only:
                return self.Z
            else:
                return phi_out / self.Z


class WeightedProjector(Gate):
    r"""
    Projector that projects the state onto an eigenvector of a Hermitian observable

    .. math::

        \rho^\prime = \frac{\Pi_n \rho \Pi_n}{Z}

    """

    def __init__(self, observable: Union[np.ndarray, tf.Tensor], wires: List[int], name=None, **kwargs):
        assert isinstance(wires, list), "'wires' must be a list"
        assert len(np.unique(wires)) == len(wires), "'wires' must be a list of unique integers"
        if name is None:
            name = 'Projector_' + '_'.join([str(s) for s in wires])
        assert len(wires) == 1, "Gate should operate on 1 qubits, found {}".format(wires)
        assert isinstance(observable,
                          (np.ndarray,
                           tf.Tensor)), f"'pi' must be of type np.ndarray or tf.Tensor, received {type(observable)}"

        assert observable.shape == (2, 2), f"pi must have shape [2,2], received {observable.shape}"
        if isinstance(observable, np.ndarray):
            self.op = tf.constant(observable, dtype=TF_COMPLEX_DTYPE)
        else:
            self.op = observable
        super(WeightedProjector, self).__init__(nparams=0, wires=wires, value=None, name=name, **kwargs)
        self.conjugate = False
        self._is_projector = True

    def build(self, input_shape):
        super(WeightedProjector, self).build(input_shape)

        einsum_indices_pi = self.ALPHABET[0] + self.ALPHABET[1]
        einsum_indices_phi = [self.ALPHABET[-1]] + [self.ALPHABET[i + 2] for i in range(len(input_shape) - 1)]
        einsum_indices_phi[self.wires[0]] = self.ALPHABET[1]
        einsum_indices_phi_out = copy.copy(einsum_indices_phi)
        einsum_indices_phi_out[self.wires[0]] = self.ALPHABET[0]
        self.einsum_contraction = ''.join(einsum_indices_pi) + ',' + ''.join(einsum_indices_phi) + '->' + ''.join(
            einsum_indices_phi_out)
        self.projectors = self.get_evals_and_projectors(self.op)[1]

    def call(self, inputs: tf.Tensor, probability_only: bool = False, **kwargs) -> tf.Tensor:
        """
        Gate Logic
        """
        if (self._batch_params):
            raise NotImplementedError

        else:
            phi_out_1 = tf.einsum(self.einsum_contraction, self.projectors[0], inputs)
            phi_out_2 = tf.einsum(self.einsum_contraction, self.projectors[1], inputs)
            self.Z1 = tf.linalg.norm(tf.reshape(phi_out_1, (-1, 1)))
            self.Z2 = tf.linalg.norm(tf.reshape(phi_out_2, (-1, 1)))
            phi_out_1 = phi_out_1 / self.Z1
            phi_out_2 = phi_out_2 / self.Z2
            self.cond = tf.reduce_all(tf.equal(tf.real(self.op), tf.eye(2, 2, dtype=TF_FLOAT_DTYPE)))
            r = tf.random.uniform([1])

            if probability_only:
                return self.Z1
            else:
                return tf.cond(tf.reduce_all(tf.equal(tf.real(self.op), tf.eye(2, 2, dtype=TF_FLOAT_DTYPE))),
                               lambda: inputs,  # if op is identity, return the state
                               lambda: tf.cond(
                                   tf.math.greater(tf.real(self.Z1), tf.constant(1e-3, dtype=TF_FLOAT_DTYPE)),
                                   # if Z larger than zero, continue
                                   lambda: tf.cond(
                                       tf.math.less(tf.real(self.Z1), tf.constant(0.999, dtype=TF_FLOAT_DTYPE)),
                                       # if Z is smaller than one, continue
                                       lambda: tf.cond(tf.math.less(tf.real(self.Z1) ** 2, r[0]),
                                                       # choose state one or two with probability Z^2
                                                       lambda: phi_out_1,
                                                       lambda: phi_out_2), lambda: phi_out_1),
                                   lambda: phi_out_2))  # else, return 1-Z^2 state

    def get_evals_and_projectors(self, O):
        """
        Get the eigenvalues and projectors of a Hermitian observable

        Args:
            *O (Tensor)*:
                Tensorflow tensor representing a Hermitian observable.

        Returns (Tuple):
            eigenvalues and the corresponding stacked projectors.

        """
        eval, evec = tf.linalg.eigh(O)
        evec = evec[:, :, tf.newaxis]
        projectors = []
        for i in range(2 ** self.nqubits):
            projectors.append(evec[:, i] * tf.transpose(evec[:, i], conjugate=True))
        return eval, tf.stack(projectors)


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
