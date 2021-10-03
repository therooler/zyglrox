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
from tensorflow import Tensor
import numpy as np
from typing import Optional, Text, List, Union, Sequence, Tuple, Any
import string
from tensorflow.python.client import device_lib
import copy


def integer_generator(start):
    """
    Generator for infinite integers. Always useful.

    Args:
        start (int):
            starting integer for the generator.

    Returns (int):
        Infinite integers.
    """
    start -= 1
    while True:
        start += 1
        yield start


def tf_kron(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    r"""
    Implementation of Kronecker product for tensorflow Tensors.

    Args:
        *a (Tensor)*:
            Tensor of size :math:`N \times M`

        *b (Tensor)*:
            Tensor of size :math:`P \times K`


    Returns (Tensor):
        Tensor of size :math:`(N \cdot P) \times (M \cdot K)`

    """
    a_shape = [a.shape[0], a.shape[1]]
    b_shape = [b.shape[0], b.shape[1]]
    return tf.reshape(tf.reshape(a, [a_shape[0], 1, a_shape[1], 1]) * tf.reshape(b, [1, b_shape[0], 1, b_shape[1]]),
                      [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]])


def partial_trace_np(psi: np.ndarray, keep: list, dims: list) -> np.ndarray:
    r"""
    Calculate the partial trace of an outer product

    .. math::

	    \rho_a = \text{Tr}_b (| u \rangle \langle u |)

    Args:
        *psi (tensor)*:
            Quantum state of shape (None ,2,2,...,2), where None is a batch dimension.

        *keep (list)*:
            An array of indices of the spaces to keep after being traced. For instance, if the space is
            A x B x C x D and we want to trace out B and D, keep = [0,2]

        *dims (list)*:
            An array of the dimensions of each space. For instance, if the space is A x B x C x D,
             dims = [None, dim_A, dim_B, dim_C, dim_D]. None is used as a batch dimension.

    Returns (Tensor):
        Partially traced out matrix

    """
    letters = string.ascii_lowercase + string.ascii_uppercase
    keep = [k + 1 for k in keep]
    assert 2 * max(keep) < len(letters) - 1, "Not enough letters for einsum..."
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = letters[-1] + ''.join([letters[i] for i in range(1, Ndim)])
    idx2 = letters[-1] + ''.join([letters[Ndim + i] if i in keep else letters[i] for i in range(1, Ndim)])
    idx_out = letters[-1] + ''.join(
        [i for i, j in zip(idx1, idx2) if i != j] + [j for i, j in zip(idx1, idx2) if i != j])
    psi = np.reshape(psi, dims)
    rho_a = np.einsum(idx1 + ',' + idx2 + '->' + idx_out, psi, np.conj(psi))
    return np.reshape(rho_a, (-1, Nkeep, Nkeep))


def partial_trace(psi: tf.Tensor, keep: list, dims: list) -> tf.Tensor:
    r"""
    Calculate the partial trace of an outer product

    .. math::

	    \rho_a = \text{Tr}_b (| u \rangle \langle u |)

    Args:
        *psi (tensor)*:
            Quantum state of shape (None ,2,2,...,2), where None is a batch dimension.

        *keep (list)*:
            An array of indices of the spaces to keep after being traced. For instance, if the space is
            A x B x C x D and we want to trace out B and D, keep = [0,2]

        *dims (list)*:
            An array of the dimensions of each space. For instance, if the space is A x B x C x D,
             dims = [None, dim_A, dim_B, dim_C, dim_D]. None is used as a batch dimension.

    Returns (Tensor):
        Partially traced out matrix

    """
    letters = string.ascii_lowercase[1:] + string.ascii_uppercase
    keep = copy.copy([k + 1 for k in keep])
    assert (len(letters) - 1) > (len(dims) - 1), "Not enough letters for einsum..."
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = 'a' + ''.join([letters[i] for i in range(1, Ndim)])
    idx2 = 'a' + ''.join([letters[Ndim + i] if i in keep else letters[i] for i in range(1, Ndim)])
    idx_out = 'a' + ''.join(
        [i for i, j in zip(idx1, idx2) if i != j] + [j for i, j in zip(idx1, idx2) if i != j])
    rho_a = tf.einsum(idx1 + ',' + idx2 + '->' + idx_out, psi, tf.math.conj(psi))
    return tf.reshape(rho_a, (-1, Nkeep, Nkeep))


def von_neumann_entropy(rho: Union[tf.Tensor, np.ndarray]) -> Union[tf.Tensor, np.ndarray]:
    r"""
    Calculate the Von Neumann entropy of a reduced density matrix.

    .. math::

	    S(\rho) = -\text{Tr} \rho \log \rho

    Args:
        *red_rho (tensor)*:
            Density matrix.

    Returns (tensor):
        Scalar containing the Von Neumann entropy.

    """
    if isinstance(rho, np.ndarray):
        rho = rho.squeeze()
        lam = np.linalg.eigvalsh(rho)
        lam = np.clip(np.real(lam), 1e-8, 1e12)
        return -np.sum(lam * np.log(lam))
    elif isinstance(rho, tf.Tensor):
        rho = tf.squeeze(rho)
        lam = tf.linalg.eigvalsh(rho)
        lam = tf.clip_by_value(tf.math.real(lam), 1e-8, 1e12)
        return -tf.reduce_sum(lam * tf.math.log(lam))
    else:
        raise ValueError("rho must be a numpy array or tf.tensor, received {}".format(type(rho)))


def logarithmic_negativity(rho: Union[tf.Tensor, np.ndarray]) -> Union[tf.Tensor, np.ndarray]:
    r"""
    Calculate the Von Neumann entropy of a reduced density matrix.

    .. math::

	    S(\rho) = -\text{Tr} \rho \log \rho

    Args:
        *red_rho (tensor)*:
            Density matrix.

    Returns (tensor):
        Scalar containing the Von Neumann entropy.

    """
    shape = rho.shape

    if isinstance(rho, np.ndarray):
        N = int(np.log2(shape[-1]))
        assert 2 ** N == shape[-1], f'Invalid rho shape {shape}'
        rho_t = np.transpose(rho.reshape(*(2 ** (N // 2) for _ in range(4))), axes=[0, 3, 2, 1])
        rho_t = rho_t.reshape((2 ** N, 2 ** N))
        evals = np.linalg.eigvalsh(rho_t)
        return np.log(1 + np.sum(np.abs(evals) - evals))
    elif isinstance(rho, tf.Tensor):
        N = int(np.log2(shape.as_list()[-1]))
        assert 2 ** N == shape[-1], f'Invalid rho shape {shape}'
        rho_t = tf.transpose(tf.reshape(rho, [2 ** (N // 2) for _ in range(4)]), perm=[0, 3, 2, 1])
        rho_t = tf.reshape(rho_t, (2 ** N, 2 ** N))
        evals = tf.math.real(tf.linalg.eigvalsh(rho_t))
        return tf.math.log(1 + tf.reduce_sum(tf.abs(evals) - evals))
    else:
        raise ValueError("rho must be a numpy array or tf.tensor, received {}".format(type(rho)))


def get_available_devices(device_type):
    assert device_type in ['CPU', 'GPU'], f"device type must be 'CPU' or 'GPU', found {device_type}"
    local_device_protos = device_lib.list_local_devices()
    return [(x.name, int(x.name.split(':')[-1])) for x in local_device_protos if x.device_type == device_type]


def renyi_entropy(rho: tf.Tensor, alpha: float = 0.5) -> tf.Tensor:
    r"""
    Calculate the Von Neumann entropy of a reduced density matrix.

    .. math::

	    S(\rho) = \frac{1}{1-\alpha}\log \text{Tr} \rho^\alpha

    Args:
        *red_rho (tensor)*:
            Density matrix.

    Returns (tensor):
        Scalar containing the Von Neumann entropy.

    """
    assert (alpha < 1) & (0 < alpha)
    if isinstance(rho, np.ndarray):
        lam = np.linalg.eigvalsh(rho)
        lam = np.clip(tf.math.real(lam), 1e-8, 1e12)
        return (1 / (1 - alpha)) * np.log(np.sum(np.power(lam, alpha), axis=1))
    elif isinstance(rho, tf.Tensor):
        lam = tf.linalg.eigvalsh(rho)
        lam = tf.clip_by_value(tf.math.real(lam), 1e-8, 1e12)
        return (1 / (1 - alpha)) * tf.math.log(tf.reduce_sum(tf.pow(lam, tf.constant(alpha, dtype=lam.dtype)), axis=1))
    else:
        raise ValueError("rho must be a numpy array or tf.tensor, received {}".format(type(rho)))


def flatten(x: tf.Tensor) -> tf.Tensor:
    """
    Flatten tensor to 1D array.

    Args:
        *x (Tensor)*:
            Input tensor with shape :math:`(M_{i_1},M_{i_2},\ldots,M_{i_m})`.

    Returns (Tensor):
        Flattened tensor with shape :math:`(\prod_n^m M_{i_n}, )`.

    """
    return tf.reshape(x, (-1,))


def ops_print(observables):
    r"""
    Print the observables in a readable manner.

    Args:
        *observables (list)*:
            List of ``Observable`` objects.

    Returns (inplace):
            None

    """
    print("List contains {} observables".format(len(observables)))
    for o in observables:
        print(o)


def tensordot(a,
              b,
              axes,
              name: Optional[Text] = None) -> Tensor:
    r"""

    Full credit for this part goes to the developers of the Google TensorNetworks library, thanks to Martin for mentioning
    this. Source: https://github.com/google/TensorNetwork/blob/master/tensornetwork/backends/tensorflow/tensordot2.py

    Tensor contraction of a and b along specified axes.
    Tensordot (also known as tensor contraction) sums the product of elements
    from `a` and `b` over the indices specified by `a_axes` and `b_axes`.
    The lists `a_axes` and `b_axes` specify those pairs of axes along which to
    contract the tensors. The axis `a_axes[i]` of `a` must have the same dimension
    as axis `b_axes[i]` of `b` for all `i` in `range(0, len(a_axes))`. The lists
    `a_axes` and `b_axes` must have identical length and consist of unique
    integers that specify valid axes for each of the tensors.
    This operation corresponds to `numpy.tensordot(a, b, axes)`.

    Example 1: When `a` and `b` are matrices (order 2), the case `axes = 1`
    is equivalent to matrix multiplication.

    Example 2: When `a` and `b` are matrices (order 2), the case
    `axes = [[1], [0]]` is equivalent to matrix multiplication.

    Example 3: Suppose that \\(a_{ijk}\\) and \\(b_{lmn}\\) represent two
    tensors of order 3. Then, `contract(a, b, [[0], [2]])` is the order 4 tensor
    \\(c_{jklm}\\) whose entry
    corresponding to the indices \\((j,k,l,m)\\) is given by:
    \\( c_{jklm} = \sum_i a_{ijk} b_{lmi} \\).

    In general, `order(c) = order(a) + order(b) - 2*len(axes[0])`.

    Args:
        *tf*:
            The TensorFlow module. This must be passed in instead of imported
        since we don't assume users have TensorFlow installed.

        *a*:
            `Tensor` of type `float32` or `float64`.

        *b*:
            `Tensor` with the same type as `a`.

        *axes*:
            Either a scalar `N`, or a list or an `int32` `Tensor` of shape [2, k].
            If axes is a scalar, sum over the last N axes of a and the first N axes of
            b in order. If axes is a list or `Tensor` the first and second row contain
            the set of unique integers specifying axes along which the contraction is
            computed, for `a` and `b`, respectively. The number of axes for `a` and
            `b` must be equal.

        *name*:
            A name for the operation (optional).

    Returns:
        A `Tensor` with the same type as `a`.

    Raises:
        ValueError:
            If the shapes of `a`, `b`, and `axes` are incompatible.

        IndexError:
            If the values in axes exceed the rank of the corresponding
            tensor.

    """

    def _tensordot_should_flip(contraction_axes: List[int],
                               free_axes: List[int]) -> bool:
        """Helper method to determine axis ordering.
        We minimize the average distance the indices would have to move under the
        transposition.
        Args:
          contraction_axes: The axes to be contracted.
          free_axes: The free axes.
        Returns:
          should_flip: `True` if `contraction_axes` should be moved to the left,
            `False` if they should be moved to the right.
        """
        # NOTE: This will fail if the arguments contain any Tensors.
        if contraction_axes and free_axes:
            return bool(np.mean(contraction_axes) < np.mean(free_axes))
        return False

    def _tranpose_if_necessary(tensor: Tensor, perm: List[int]) -> Tensor:
        """Like transpose(), but avoids creating a new tensor if possible.
        Although the graph optimizer should kill trivial transposes, it is best not
        to add them in the first place!
        """
        if perm == list(range(len(perm))):
            return tensor
        return tf.transpose(tensor, perm)

    def _reshape_if_necessary(tensor: Tensor,
                              new_shape: List[int]) -> Tensor:
        """Like reshape(), but avoids creating a new tensor if possible.
        Assumes shapes are both fully specified."""
        cur_shape = tensor.get_shape().as_list()
        if (len(new_shape) == len(cur_shape) and
                all(d0 == d1 for d0, d1 in zip(cur_shape, new_shape))):
            return tensor
        return tf.reshape(tensor, new_shape)

    def _tensordot_reshape(
            a: Tensor, axes: Union[Sequence[int], Tensor], is_right_term=False
    ) -> Tuple[Tensor, Union[List[int], Tensor], Optional[List[int]], bool]:
        """Helper method to perform transpose and reshape for contraction op.
        This method is helpful in reducing `math_ops.tensordot` to `math_ops.matmul`
        using `array_ops.transpose` and `array_ops.reshape`. The method takes a
        tensor and performs the correct transpose and reshape operation for a given
        set of indices. It returns the reshaped tensor as well as a list of indices
        necessary to reshape the tensor again after matrix multiplication.
        Args:
          a: `Tensor`.
          axes: List or `int32` `Tensor` of unique indices specifying valid axes of
           `a`.
          is_right_term: Whether `a` is the right (second) argument to `matmul`.
        Returns:
          A tuple `(reshaped_a, free_dims, free_dims_static, transpose_needed)`
          where `reshaped_a` is the tensor `a` reshaped to allow contraction via
          `matmul`, `free_dims` is either a list of integers or an `int32`
          `Tensor`, depending on whether the shape of a is fully specified, and
          free_dims_static is either a list of integers and None values, or None,
          representing the inferred static shape of the free dimensions.
          `transpose_needed` indicates whether `reshaped_a` must be transposed,
          or not, when calling `matmul`.
        """
        if a.get_shape().is_fully_defined() and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            # NOTE: This will fail if axes contains any tensors
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            flipped = _tensordot_should_flip(axes, free)

            free_dims = [shape_a[i] for i in free]
            prod_free = int(np.prod([shape_a[i] for i in free]))
            prod_axes = int(np.prod([shape_a[i] for i in axes]))
            perm = axes + free if flipped else free + axes
            new_shape = [prod_axes, prod_free] if flipped else [prod_free, prod_axes]
            transposed_a = _tranpose_if_necessary(a, perm)
            reshaped_a = _reshape_if_necessary(transposed_a, new_shape)
            transpose_needed = (not flipped) if is_right_term else flipped
            return reshaped_a, free_dims, free_dims, transpose_needed
        if a.get_shape().ndims is not None and isinstance(axes, (list, tuple)):
            shape_a = a.get_shape().as_list()
            axes = [i if i >= 0 else i + len(shape_a) for i in axes]
            free = [i for i in range(len(shape_a)) if i not in axes]
            flipped = _tensordot_should_flip(axes, free)
            perm = axes + free if flipped else free + axes

            axes_dims = [shape_a[i] for i in axes]
            free_dims = [shape_a[i] for i in free]
            free_dims_static = free_dims
            axes = tf.convert_to_tensor(axes, dtype=tf.dtypes.int32, name="axes")
            free = tf.convert_to_tensor(free, dtype=tf.dtypes.int32, name="free")
            shape_a = tf.shape(a)
            transposed_a = _tranpose_if_necessary(a, perm)
        else:
            free_dims_static = None
            shape_a = tf.shape(a)
            rank_a = tf.rank(a)
            axes = tf.convert_to_tensor(axes, dtype=tf.dtypes.int32, name="axes")
            axes = tf.where(axes >= 0, axes, axes + rank_a)
            free, _ = tf.compat.v1.setdiff1d(tf.range(rank_a), axes)
            # Matmul does not accept tensors for its transpose arguments, so fall
            # back to the previous, fixed behavior.
            # NOTE(amilsted): With a suitable wrapper for `matmul` using e.g. `case`
            #   to match transpose arguments to tensor values, we could also avoid
            #   unneeded tranposes in this case at the expense of a somewhat more
            #   complicated graph. Unclear whether this would be beneficial overall.
            flipped = is_right_term
            perm = (
                tf.concat([axes, free], 0) if flipped else tf.concat([free, axes], 0))
            transposed_a = tf.transpose(a, perm)

        free_dims = tf.gather(shape_a, free)
        axes_dims = tf.gather(shape_a, axes)
        prod_free_dims = tf.reduce_prod(free_dims)
        prod_axes_dims = tf.reduce_prod(axes_dims)

        if flipped:
            new_shape = tf.stack([prod_axes_dims, prod_free_dims])
        else:
            new_shape = tf.stack([prod_free_dims, prod_axes_dims])
        reshaped_a = tf.reshape(transposed_a, new_shape)
        transpose_needed = (not flipped) if is_right_term else flipped
        return reshaped_a, free_dims, free_dims_static, transpose_needed

    def _tensordot_axes(a: Tensor, axes
                        ) -> Tuple[Any, Any]:
        """Generates two sets of contraction axes for the two tensor arguments."""
        a_shape = a.get_shape()
        if isinstance(axes, tf.compat.integral_types):
            if axes < 0:
                raise ValueError("'axes' must be at least 0.")
            if a_shape.ndims is not None:
                if axes > a_shape.ndims:
                    raise ValueError("'axes' must not be larger than the number of "
                                     "dimensions of tensor %s." % a)
                return (list(range(a_shape.ndims - axes,
                                   a_shape.ndims)), list(range(axes)))
            rank = tf.rank(a)
            return (tf.range(rank - axes, rank,
                             dtype=tf.int32), tf.range(axes, dtype=tf.int32))
        if isinstance(axes, (list, tuple)):
            if len(axes) != 2:
                raise ValueError("'axes' must be an integer or have length 2.")
            a_axes = axes[0]
            b_axes = axes[1]
            if isinstance(a_axes, tf.compat.integral_types) and \
                    isinstance(b_axes, tf.compat.integral_types):
                a_axes = [a_axes]
                b_axes = [b_axes]
            # NOTE: This fails if either a_axes and b_axes are Tensors.
            if len(a_axes) != len(b_axes):
                raise ValueError(
                    "Different number of contraction axes 'a' and 'b', %s != %s." %
                    (len(a_axes), len(b_axes)))

            # The contraction indices do not need to be permuted.
            # Sort axes to avoid unnecessary permutations of a.
            # NOTE: This fails if either a_axes and b_axes contain Tensors.
            # pylint: disable=len-as-condition
            if len(a_axes) > 0:
                a_axes, b_axes = list(zip(*sorted(zip(a_axes, b_axes))))

            return a_axes, b_axes
        axes = tf.convert_to_tensor(axes, name="axes", dtype=tf.int32)
        return axes[0], axes[1]

    with tf.compat.v1.name_scope(name, "Tensordot", [a, b, axes]) as _name:
        a = tf.convert_to_tensor(a, name="a")
        b = tf.convert_to_tensor(b, name="b")
        a_axes, b_axes = _tensordot_axes(a, axes)
        a_reshape, a_free_dims, a_free_dims_static, a_transp = _tensordot_reshape(
            a, a_axes)
        b_reshape, b_free_dims, b_free_dims_static, b_transp = _tensordot_reshape(
            b, b_axes, is_right_term=True)

        ab_matmul = tf.matmul(
            a_reshape, b_reshape, transpose_a=a_transp, transpose_b=b_transp)

        if isinstance(a_free_dims, list) and isinstance(b_free_dims, list):
            return tf.reshape(ab_matmul, a_free_dims + b_free_dims, name=_name)
        a_free_dims = tf.convert_to_tensor(a_free_dims, dtype=tf.dtypes.int32)
        b_free_dims = tf.convert_to_tensor(b_free_dims, dtype=tf.dtypes.int32)
        product = tf.reshape(
            ab_matmul, tf.concat([a_free_dims, b_free_dims], 0), name=_name)
        if a_free_dims_static is not None and b_free_dims_static is not None:
            product.set_shape(a_free_dims_static + b_free_dims_static)
        return product
