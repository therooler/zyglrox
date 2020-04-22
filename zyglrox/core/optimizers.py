import tensorflow as tf
from tensorflow.python.ops.parallel_for.gradients import jacobian
import numpy as np


def Newton(loss, vrs, optimizer=None):
    """
    Newton's second order gradient method.

    Args:

        *loss (Tensor)*:
            Loss function that uses the provided state.

        *vrs (Variables)*:
            Variables to be optimized.

        *optimizer (Tensorflow Optimizer)*:
            Tensorflow optimizer from the tf.train API

    Returns (Operation):
        Train step operation.

    """
    nparams = prep_variables(vrs)

    if optimizer == None:
        optimizer = tf.train.GradientDescentOptimizer(1.0)

    grads = tf.gradients(loss, vrs)
    hessian = tf.hessians(loss, vrs)
    # if len(vrs) == 1:
    #     grads = tf.reshape(grads, ([1] + [nparams]))
    #     grads = tf.split(grads, nparams, axis=1)
    #
    #
    # hessian = [tf.gradients(g, vrs) for g in grads]
    # print(hessian)
    # # hessian = tf.stack([tf.gradients(g, vrs, stop_gradients=loss) for g in grads])
    hessian = tf.reshape(hessian, (nparams, nparams))

    # grads = tf.stack(grads)
    grads = tf.linalg.solve(hessian, tf.reshape(grads, (-1, 1)))

    if len(vrs)==1:
        grads = tf.reshape(grads, (vrs[0].shape))
        return optimizer.apply_gradients(zip([grads], vrs)), grads
    else:
        grads = tf.split(grads, nparams, axis=0)
        return optimizer.apply_gradients(zip(grads, vrs))


def ImaginaryTimeEvolution(state, loss, vrs, optimizer=None):
    """
    Implementation of the imaginary time evolution gradient method

    Args:
        *state (Tensor)*:
            Output state of the circuit.

        *loss (Tensor)*:
            Loss function that uses the provided state.

        *vrs (Variables)*:
            Variables to be optimized.

        *optimizer (Tensorflow Optimizer)*:
            Tensorflow optimizer from the tf.train API

    Returns (Operation):
        Train step operation.

    """
    if optimizer == None:
        optimizer = tf.train.GradientDescentOptimizer(0.01)
    for variable in vrs:
        jac, nparams = prepGeometricTensor(state, variable)

        grads = tf.gradients(loss, variable)
        sf_metric = []
        for i in range(nparams):
            for j in range(nparams):
                part_1 = tf.math.conj(tf.reshape(jac[i], (1, -1))) @ tf.reshape(jac[j], (-1, 1))
                sf_metric.append(part_1)
        eta = tf.math.real(tf.reshape(tf.stack(sf_metric), (nparams, nparams)))
        grads = tf.stack(grads)
        grads = tf.linalg.solve(eta, tf.reshape(grads, (-1, 1)))

        if len(variable)==1:
            grads = tf.reshape(grads, (variable[0].shape))
            return optimizer.apply_gradients(zip([grads], variable))
        else:
            grads = tf.split(grads, nparams, axis=0)
            return optimizer.apply_gradients(zip(grads, variable))

def QuantumNaturalGradient(state, loss, vrs, optimizer=None, stability_shift = None):
    """
    Implementation of the quantum natural gradient gradient method

    Args:
        *state (Tensor)*:
            Output state of the circuit.

        *loss (Tensor)*:
            Loss function that uses the provided state.

        *vrs (Variables)*:
            Variables to be optimized.

        *optimizer (Tensorflow Optimizer)*:
            Tensorflow optimizer from the tf.train API

    Returns (Operation):
        Train step operation.

    """
    if optimizer == None:
        optimizer = tf.train.GradientDescentOptimizer(0.01)
    assign_ops = []
    for variable in vrs:
        variable = [variable]
        jac, nparams = prepGeometricTensor(state, variable)

        grads = tf.gradients(loss, variable)
        sf_metric = []

        for i in range(nparams):
            for j in range(nparams):
                part_1 = tf.math.conj(tf.reshape(jac[i], (1, -1))) @ tf.reshape(jac[j], (-1, 1))
                part_2 = tf.math.conj(tf.reshape(jac[i], (1, -1))) @ tf.reshape(state, (-1, 1)) +\
                         tf.math.conj(tf.reshape(state, (1, -1))) @ tf.reshape(jac[j], (-1, 1))
                sf_metric.append(part_1 - part_2)
        eta = tf.math.real(tf.reshape(tf.stack(sf_metric), (nparams, nparams)))
        grads = tf.stack(grads)
        if stability_shift is not None:
            eta += tf.eye(*eta.shape.as_list()) * stability_shift
        grads = tf.linalg.solve(eta, tf.reshape(grads, (-1, 1)))
        if len(variable)==1:
            grads = tf.reshape(grads, (variable[0].shape))
            assign_ops.append(optimizer.apply_gradients(zip([grads], variable)))
        else:
            grads = tf.split(grads, nparams, axis=0)
            assign_ops.append(optimizer.apply_gradients(zip(grads, variable)))
    return assign_ops

def prepGeometricTensor(state, vrs):
    nparams = prep_variables(vrs)
    phi_r = tf.math.real(state)
    phi_c = tf.math.imag(state)
    jac_r = jacobian(phi_r, vrs)
    jac_c = jacobian(phi_c, vrs)
    if len(vrs) == 1:
        jac = tf.reshape(tf.complex(jac_r, jac_c), (state.shape + [nparams]))
        jac = tf.split(jac, nparams, axis=-1)
        jac = [tf.reshape(v, state.shape) for v in jac]
    else:
        jac = [tf.complex(jac_r[i], jac_c[i]) for i in range(nparams)]
    return jac, nparams

def prep_variables(vrs):
    assert isinstance(vrs, list), "vrs must be a list, received {}".format(type(vrs))
    assert len(vrs) > 0, "vrs is an empty list"
    if len(vrs) == 1:
        nparams = np.prod(vrs[0].shape)
    else:
        assert all(v.shape == (1, 1) for v in
                   vrs), "expected list of single Variable or list of only variables with shape (1,1), received {}".format(
            vrs)
        nparams = len(vrs)
    return nparams