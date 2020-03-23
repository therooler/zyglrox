import tensorflow as tf
import warnings

TF_FLOAT_DTYPE = tf.float32
TF_COMPLEX_DTYPE = tf.complex64


def set_dtype(global_name: str, dtype: str):
    global TF_FLOAT_DTYPE
    global TF_COMPLEX_DTYPE
    possible_global_dtypes = ['TF_FLOAT_DTYPE', 'TF_COMPLEX_DTYPE']
    possible_dtypes = ['tf.float32', 'tf.float64', 'tf.complex64', 'tf.complex128']
    dtype_dict = {'tf.float32': tf.float32, 'tf.float64': tf.float64, 'tf.complex64': tf.complex64,
                  'tf.complex128': tf.complex128}
    assert global_name in possible_global_dtypes, "name must be in {}, received {}".format(possible_global_dtypes,
                                                                                           global_name)
    assert dtype in possible_dtypes, "dtype must be in {}, received {}".format(possible_global_dtypes, global_name)
    if global_name == 'TF_FLOAT_DTYPE':
        TF_FLOAT_DTYPE = dtype_dict[dtype]
    elif global_name == 'TF_COMPLEX_DTYPE':
        TF_COMPLEX_DTYPE = dtype_dict[dtype]

    if dtype == 'TF_COMPLEX_DTYPE':
        if TF_FLOAT_DTYPE.name == tf.float32.name:
            if TF_FLOAT_DTYPE.name == tf.complex64.name:
                warnings.warn("For float type tf.float32, the corresponding complex type must be tf.complex64")
        if TF_FLOAT_DTYPE.name == tf.float64.name:
            if TF_COMPLEX_DTYPE.name == tf.complex128.name:
                warnings.warn("For float type tf.float64, the corresponding complex type must be tf.complex128")
        else:
            raise ValueError('TF_FLOAT_DTYPE must be either tf.float32 or tf.float64, received {}'.format(TF_FLOAT_DTYPE))
