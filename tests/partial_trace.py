from zyglrox.core.utils import partial_trace
import tensorflow as tf
import numpy as np

# Phi+ state
psi = tf.reshape(tf.constant(1 / np.sqrt(2) * np.array([1,0,0,1])), (1,2,2))
rho_a = partial_trace(psi, keep=[0,], dims=[1,2,2])

sess = tf.Session()
print(sess.run(rho_a))