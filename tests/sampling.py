from zyglrox.core.observables import Observable, SampleExpectationValue, ExpectationValue
from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.gates import *
import tensorflow as tf

def sampling_single_qubit():

    qc = QuantumCircuit(1, [Hadamard(wires=[0, ])], device='CPU')
    obs = [Observable("x", wires=[0, ]), Observable("z", wires=[0, ])]
    phi = qc.circuit(qc.phi)

    expval_layer = SampleExpectationValue(obs)
    qc.initialize()
    expvals = qc._sess.run(expval_layer(phi))
    print(expvals)

def sampling_double_qubit():

    qc = QuantumCircuit(2, [Hadamard(wires=[0, ]), Hadamard(wires=[1, ])], device='CPU')
    obs = [Observable("x", wires=[0, ]), Observable("z", wires=[0, ]),Observable("x", wires=[1, ]), Observable("z", wires=[1, ])]
    phi = qc.circuit(qc.phi)
    qc.initialize()
    expval_layer = SampleExpectationValue(obs, number_of_samples=200)
    expvals = qc._sess.run(expval_layer(phi))
    print(expvals.shape)
    for e in expvals:
        print(e)


if __name__=="__main__":
    sampling_single_qubit()