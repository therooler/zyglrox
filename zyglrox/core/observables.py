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
import tensorflow_probability as tfp

from typing import List
from tensorflow.keras.layers import Layer
from zyglrox.core.gates import identity, pauli_x, pauli_y, pauli_z, hermitian
from zyglrox.core.utils import tensordot
from zyglrox.core._config import TF_FLOAT_DTYPE, TF_COMPLEX_DTYPE
import numpy as np
from inspect import signature
from tensorflow.python.framework import tensor_shape

observable_dict = {'I': identity, 'x': pauli_x, 'y': pauli_y, 'z': pauli_z, 'hermitian': hermitian}


# observable_dict = {'I': identity, 'X': pauli_x, 'Y': pauli_y, 'Z': pauli_z, 'Hermitian': hermitian}


class Observable(Layer):
    """
    Abstract Observable class

    """

    def __init__(self, op_name: str, wires: List[int], **kwargs):
        """
        The ``Observable`` class calculates Hermitian observables from the quantum circuit.

        Args:
            *op_name (str)*:
                Name of an operation defined in the ``observable_dict``

            *wires (list)*:
                List of numbers where the observable of interest is to be measured

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """
        super(Observable, self).__init__(
            name='observable_{}_{}'.format(kwargs['name'] if 'name' in kwargs else op_name,
                                           ''.join([str(s) for s in wires])))
        if op_name.lower() in ['x', 'y', 'z', 'hermitian']:
            op_name = op_name.lower()
        assert op_name in observable_dict.keys(), "Operation {} not recognized, choose from {}".format(op_name,
                                                                                                       list(
                                                                                                           observable_dict.keys()))
        self.op_name = op_name
        # shift all dimensions due to batch dim
        self.wires = [w + 1 for w in wires]
        self.sample = kwargs.pop('sample', False)
        self.number_of_samples = kwargs.pop('number_of_samples', 100)
        self.value = kwargs.pop('value', None)
        self.nqubits = len(self.wires)

    def __str__(self):
        """

        Prints name of the ``Observable`` object

        """
        return self.name

    def build(self, input_shape, **kwargs):
        r"""
        Called once from `__call__`, when we know the shapes of inputs and `dtype`.
        Should initialize the trainable variables, and call the super's `build()`.

        Args:
            *input_shape (list)*:
                Input shapes of the incoming tensor.

        """
        super(Observable, self).build(input_shape)
        assert len(input_shape) > 1, "Input shape must have at least dim>1, received {}".format(input_shape)

        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if None in input_shape:
            assert ((input_shape[0] == None)) & (
                    input_shape.count(None) == 1), "Only the first dimension can be None, found {}".format(
                input_shape)
        elif (input_shape[0] > 1):
            assert all([input_shape[i] == 2 for i in range(1, len(input_shape))]), \
                "Input vector needs to have shape (x, 2, 2,...,2), found {}".format(input_shape)
        else:
            assert (input_shape[0] == 1) & (all([input_shape[i] == 2 for i in range(1, len(input_shape))])), \
                "Input vector needs to have shape (1, 2, 2,...,2), found {}".format(input_shape)
        self.total_qubits = len(input_shape) - 1
        operation = observable_dict[self.op_name]
        sig = signature(operation)
        if (len(sig.parameters) > 0) & (self.value is not None):
            operation = operation(self.value)
        else:
            operation = operation()
        self.evals, self.projectors = self.get_evals_and_projectors(operation)
        self.projectors = tf.reshape(self.projectors, shape=[2 ** self.nqubits] + [2] * self.nqubits * 2)
        self.evals = tf.reshape(self.evals, shape=[1, 1] + [2 ** self.nqubits])

        self.operation = tf.reshape(operation, shape=[2] * self.nqubits * 2)

        assert max(
            self.wires) - 1 < self.total_qubits, "Operation {} is performed on qubit {}, but system only has {} qubits".format(
            self.op_name, max(self.wires) - 1, self.nqubits)

        if self.sample:
            unused_idxs = [idx for idx in range(2, self.total_qubits + 2) if idx not in [w + 1 for w in self.wires]]
            perm = [0] + [w + 1 for w in self.wires] + [1] + unused_idxs
            self.inv_perm = np.argsort(perm)
        else:
            unused_idxs = [idx for idx in range(1, self.total_qubits + 1) if idx not in self.wires]
            perm = self.wires + [0] + unused_idxs
            self.inv_perm = np.argsort(perm)

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

    def call(self, inputs, **kwargs):
        r"""
        Called in the Keras Model ``__call__`` method after making sure ``build()`` has been called once. Should actually
        perform the logic of applying the layer to the input tensors (which should be passed in as the first argument).

        Args:
            *inputs (Tensor)*:
                Input tensor corresponding to the wave function

            *\*\*kwargs*:
                Additional arguments.

        Returns (Tensor):
            Output tensor corresponding to wave function after the circuit has been applied

        """
        if self.sample:
            phi = tensordot(self.projectors, inputs,
                            axes=[list(range(self.nqubits + 1, 2 * self.nqubits + 1)), self.wires])
            phi = tf.transpose(phi, perm=self.inv_perm)
            probs = tf.math.conj(tf.reshape(inputs, (-1, 1, int(2 ** self.total_qubits)))) @ tf.reshape(phi, (
                -1, int(2 ** self.total_qubits), 1))
            probs = tf.reshape(probs, (-1, 2 ** self.nqubits))
            sampler = tfp.distributions.OneHotCategorical(probs=tf.math.real(probs))
            samples = tf.cast(sampler.sample(self.number_of_samples), TF_COMPLEX_DTYPE)
            obs = tf.reduce_mean(tf.reduce_sum(samples * self.evals, axis=2), axis=0)
            obs = tf.reshape(obs, (-1, 1, 1))
        else:
            phi = tensordot(self.operation, inputs, axes=[list(range(self.nqubits, 2 * self.nqubits)), self.wires])
            phi = tf.transpose(phi, perm=self.inv_perm)
            obs = tf.math.conj(tf.reshape(inputs, (-1, 1, int(2 ** self.total_qubits)))) @ tf.reshape(phi, (
                -1, int(2 ** self.total_qubits), 1))
        return tf.math.real(obs)
        # return obs


class ExpectationValue(Layer):
    """
    Abstract Observable class

    """

    def __init__(self, observables):
        """
        The ``ExpectationValue`` class calculates batches of Hermitian observables from the quantum circuit.

        Args:
            *observables (str)*:
                List of ``Observable`` objects

        Returns (inplace):
            None

        """
        super(ExpectationValue, self).__init__(name='expectation_value')

        self.observables = observables

    def __str__(self):
        """

        Prints name of the ``ExpectationValue`` object

        """
        return self.name

    def build(self, input_shape):
        r"""
        Called once from `__call__`, when we know the shapes of inputs and `dtype`.
        Should initialize the trainable variables, and call the super's `build()`.

        Args:
            *input_shape (list)*:
                Input shapes of the incoming tensor.

        """
        super(ExpectationValue, self).build(input_shape)
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if None in input_shape:
            assert ((input_shape[0] == None)) & (
                    input_shape.count(None) == 1), "Only the first dimension can be None, found {}".format(
                input_shape)
        elif (input_shape[0] > 1):
            assert all([input_shape[i] == 2 for i in range(1, len(input_shape))]), \
                "Input vector needs to have shape (x, 2, 2,...,2), found {}".format(input_shape)
        else:
            assert (input_shape[0] == 1) & (all([input_shape[i] == 2 for i in range(1, len(input_shape))])), \
                "Input vector needs to have shape (1, 2, 2,...,2), found {}".format(input_shape)

    def call(self, inputs, **kwargs):
        r"""
        Called in the Keras Model ``__call__`` method after making sure ``build()`` has been called once. Should actually
        perform the logic of applying the layer to the input tensors (which should be passed in as the first argument).

        Args:
            *inputs (Tensor)*:
                Input tensor corresponding to the wave function

            *\*\*kwargs*:
                Additional arguments.

        Returns (Tensor):
            Batch of expectation values of shape (None, n_observables,1,1)

        """
        expvals = []
        # loop through all the observables and calculate their expectation values
        for obs in self.observables:
            expvals.append(obs(inputs))
        # stack the expectation values into a single tensor, put the batch dimension first.
        return tf.transpose(tf.stack(expvals), perm=[1, 0, 2, 3])


class SampleExpectationValue(Layer):
    """
    Abstract Observable class

    """

    def __init__(self, observables: List[Observable], number_of_samples: int = 100):
        """
        The ``ExpectationValue`` class calculates batches of Hermitian observables from the quantum circuit.

        Args:
            *observables (str)*:
                List of ``Observable`` objects


            *number_of_samples (int)*:
                The number of samples used for determining the observation values.

        Returns (inplace):
            None

        """
        super(SampleExpectationValue, self).__init__(name='expectation_value')

        self.observables = observables
        for obs in self.observables:
            obs.sample = True
            obs.number_of_samples = number_of_samples

    def __str__(self):
        """

        Prints name of the ``ExpectationValue`` object

        """
        return self.name

    def build(self, input_shape):
        r"""
        Called once from `__call__`, when we know the shapes of inputs and `dtype`.
        Should initialize the trainable variables, and call the super's `build()`.

        Args:
            *input_shape (list)*:
                Input shapes of the incoming tensor.

        """
        super(SampleExpectationValue, self).build(input_shape)
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if None in input_shape:
            assert ((input_shape[0] == None)) & (
                    input_shape.count(None) == 1), "Only the first dimension can be None, found {}".format(
                input_shape)
        elif (input_shape[0] > 1):
            assert all([input_shape[i] == 2 for i in range(1, len(input_shape))]), \
                "Input vector needs to have shape (x, 2, 2,...,2), found {}".format(input_shape)
        else:
            assert (input_shape[0] == 1) & (all([input_shape[i] == 2 for i in range(1, len(input_shape))])), \
                "Input vector needs to have shape (1, 2, 2,...,2), found {}".format(input_shape)

    def call(self, inputs, **kwargs):
        r"""
        Called in the Keras Model ``__call__`` method after making sure ``build()`` has been called once. Should actually
        perform the logic of applying the layer to the input tensors (which should be passed in as the first argument).

        Args:
            *inputs (Tensor)*:
                Input tensor corresponding to the wave function

            *\*\*kwargs*:
                Additional arguments.

        Returns (Tensor):
            Batch of expectation values of shape (None, n_observables,1,1)

        """
        expvals = []
        # loop through all the observables and calculate their expectation values
        for obs in self.observables:
            expvals.append(obs(inputs))
        # stack the expectation values into a single tensor, put the batch dimension first.
        return tf.transpose(tf.stack(expvals), perm=[1, 0, 2, 3])
