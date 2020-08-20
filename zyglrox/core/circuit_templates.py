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
from zyglrox.core.gates import *
from zyglrox.core._config import TF_FLOAT_DTYPE


def verify_circuit_input(N, depth, initial_parameters, desired_shape):
    assert isinstance(N, int), "N must be an integer, received {}".format(type(N))
    assert isinstance(depth, int), "depth must be an integer, received {}".format(type(depth))
    assert initial_parameters.shape.as_list() == list(desired_shape), "Expected array of shape {}, " \
                                                      "received {}".format(desired_shape,
                                                                           initial_parameters.shape)


def staircase_cnot(N: int, mod=True):
    r"""
    Circuit architecture where :math:`N` CNOTs are applied to qubit :math:`i` with target :math:`(i+1)\text{mod} (N)`

    #TODO: insert figure of circuit

    Args:
        *N (int)*:
            Number of qubits

    Returns (list):
        List of ``Gate`` objects.

    """
    gates = []
    for i in range(N - 1):
        gates.append(CNOT(wires=[i, i + 1]))
    if mod:
        gates.append(CNOT(wires=[N - 1, 0]))
    return gates


def target_staircase_cnot(N: int, target: int):
    r"""
    Circuit architecture where :math:`N-1` CNOTs are applied, all with the same target qubit.

    #TODO: insert figure of circuit

    Args:
        *N (int)*:
            Number of qubits.

        *target (int)*:
            Target qubit in the staircase circuit.

    Returns (list):
        List of ``Gate`` objects.

    """
    gates = []
    for i in range(N):
        if i != target:
            gates.append(CNOT(wires=[i, target]))
    return gates


def circuit6(N: int):
    """
    The most expressive circuit at depth :math:`L=1` according to `Sim et al. (2019) <https://arxiv.org/abs/1905.10876>`_.

    Args:
        *N (int)*:
            Number of qubits.

    Returns (list):
        List of ``Gate`` objects.

    """
    gates = []
    for i in range(N):
        gates.append(RX(wires=[i]))
    for i in range(N):
        gates.append(RZ(wires=[i]))

    for ctrl in range(N):
        for i in range(N):
            if i != ctrl:
                gates.append(CNOT(wires=[ctrl, i]))

    for i in range(N):
        gates.append(RX(wires=[i]))
    for i in range(N):
        gates.append(RZ(wires=[i]))
    return gates


def custom_cnot(locs: List):
    """
    Make custom CNOT padding

    Args:
        *locs (list)*:
            List with tuples with (control, target).

    Returns (list):
        List of ``Gate`` objects.

    """
    gates = []
    for i, j in locs:
        gates.append(CNOT(wires=[i, j]))
    return gates


def tfi_1d_hva_circuit(N: int, depth: int, initial_parameters):
    verify_circuit_input(N, depth, initial_parameters, (2, depth))

    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))
    gates = []

    gates.extend([Hadamard(wires=[i, ]) for i in range(N)])
    for l in range(depth):
        for j in range(0, N - 1, 2):
            gates.append(ZZ(wires=[j, j + 1], value=parameters[0, l]))

        for j in range(1, N - 1, 2):
            gates.append(ZZ(wires=[j, j + 1], value=parameters[0, l]))

        gates.append(ZZ(wires=[N - 1, 0], value=parameters[0, l]))
        gates.extend([RX(wires=[i, ], value=parameters[1, l]) for i in range(N)])
    return gates


def tfi_2d_hva_circuit(N: int, depth: int, edge_coloring: dict, initial_parameters):
    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))
    gates = []

    gates.extend([Hadamard(wires=[i]) for i in range(N)])

    for d in range(depth):
        for alpha, graph in enumerate(edge_coloring.values()):
            gates.extend([ZZ(wires=edge, value=parameters[0, d]) for edge in graph])
        gates.extend([RX(wires=[i], value=parameters[1, d]) for i in range(N)])

    return gates


def xxz_1d_hva_circuit(N: int, depth: int, initial_parameters):
    verify_circuit_input(N, depth, initial_parameters, (2, 2, depth))
    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))
    gates = []
    for i in range(0, N - 1, 2):
        gates.append(PauliX(wires=[i]))
        gates.append(PauliX(wires=[i + 1]))
        gates.append(Hadamard(wires=[i]))
        gates.append(CNOT(wires=[i, i + 1]))

    for l in range(depth):
        for j in range(1, N - 1, 2):
            gates.append(ZZ(wires=[j, j + 1], value=parameters[0, 0, l]))
            gates.append(YY(wires=[j, j + 1], value=parameters[0, 1, l]))
            gates.append(XX(wires=[j, j + 1], value=parameters[0, 1, l]))
        gates.append(ZZ(wires=[N - 1, 0], value=parameters[0, 0, l]))
        gates.append(YY(wires=[N - 1, 0], value=parameters[0, 1, l]))
        gates.append(XX(wires=[N - 1, 0], value=parameters[0, 1, l]))

        for j in range(0, N - 1, 2):
            gates.append(ZZ(wires=[j, j + 1], value=parameters[1, 0, l]))
            gates.append(YY(wires=[j, j + 1], value=parameters[1, 1, l]))
            gates.append(XX(wires=[j, j + 1], value=parameters[1, 1, l]))

    return gates


def xxz_2d_hva_circuit(depth: int, edge_coloring: dict, initial_parameters: np.ndarray):
    subgraphs = list(edge_coloring.values())
    gates = []

    # prepare ground state here
    for d in subgraphs[-1]:
        # Bellstates
        gates.append(PauliX(wires=[d[0]]))
        gates.append(PauliX(wires=[d[1]]))
        gates.append(Hadamard(wires=[d[0]]))
        gates.append(CNOT(wires=[d[0], d[1]]))

    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))

    for d in range(depth):
        for alpha, graph in enumerate(subgraphs):
            gates.extend([XX(wires=edge, value=parameters[alpha, 0, d]) for edge in graph])
            gates.extend([YY(wires=edge, value=parameters[alpha, 1, d]) for edge in graph])
            gates.extend([ZZ(wires=edge, value=parameters[alpha, 1, d]) for edge in graph])
    return gates


def kitaev_honeycomb_circuit(depth: int, edge_coloring: dict, initial_parameters: np.ndarray):
    gates = []
    print(edge_coloring)
    # prepare ground state here
    for edge in edge_coloring['yy']:
        # Bellstates
        gates.append(PauliX(wires=[edge[0]]))
        gates.append(PauliX(wires=[edge[1]]))
        gates.append(Hadamard(wires=[edge[0]]))
        gates.append(CNOT(wires=[edge[0], edge[1]]))

    parameters = tf.Variable(tf.constant(initial_parameters, TF_FLOAT_DTYPE),
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))

    for d in range(depth):
        for interaction in ['zz', 'yy', 'xx']:
            graph = edge_coloring[interaction]
            if interaction == 'xx':
                gates.extend([XX(wires=edge, value=parameters[0, d]) for edge in graph])
            if interaction == 'yy':
                gates.extend([YY(wires=edge, value=parameters[1, d]) for edge in graph])
            if interaction == 'zz':
                gates.extend([ZZ(wires=edge, value=parameters[2, d]) for edge in graph])
    return gates
