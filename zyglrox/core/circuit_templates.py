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
from zyglrox.core.circuit import QuantumCircuit
from zyglrox.core.utils import get_available_devices
import itertools as it


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


def tfi_1d_hva_circuit(N: int, depth: int, initial_parameters, boundary_condition: str):
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

        if boundary_condition == 'closed':
            gates.append(ZZ(wires=[N - 1, 0], value=parameters[0, l]))
        # elif boundary_condition=='open':
        gates.extend([RX(wires=[i, ], value=parameters[1, l]) for i in range(N)])
    return gates


def tfi_1d_projective_hva_circuit(N: int, depth: int, projectors, initial_parameters, boundary_condition: str, nmeas:int):
    verify_circuit_input(N, depth, initial_parameters, (2, depth))

    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))

    projector_combinations = list(it.product((0, 1), repeat=nmeas))
    projective_circuits = []
    gpus = get_available_devices('GPU')

    for num, comb in enumerate(projector_combinations):
        gpu_num = int(num % len(gpus))

        projectors_comb = [projectors[i] for i in comb]
        gates_proj = []
        gates_proj.extend([Hadamard(wires=[i, ]) for i in range(N)])
        for l in range(depth):
            for j in range(0, N - 1, 2):
                gates_proj.append(ZZ(wires=[j, j + 1], value=parameters[0, l]))
            for j in range(1, N - 1, 2):
                gates_proj.append(ZZ(wires=[j, j + 1], value=parameters[0, l]))
            if boundary_condition == 'closed':
                gates_proj.append(ZZ(wires=[N - 1, 0], value=parameters[0, l]))
            gates_proj.append(Projector(pi=projectors_comb[l], wires=[l, ]))
            for i in range(N):
                gates_proj.append(RX(wires=[i, ], value=parameters[1, l]))
        projective_circuits.append([QuantumCircuit(nqubits=N, gates=gates_proj, device_number=gpu_num, device='GPU'), ])
        projective_circuits[num][0].execute()
        projective_circuits[num].append(tf.math.real(
            tf.reduce_prod(
                tf.stack([l.Z for l in projective_circuits[num][0].circuit.layers if l._is_projector]))) ** 2)

    gates_unit = []

    gates_unit.extend([Hadamard(wires=[i, ]) for i in range(N)])
    for l in range(depth):
        for j in range(0, N - 1, 2):
            gates_unit.append(ZZ(wires=[j, j + 1], value=parameters[0, l]))
        for j in range(1, N - 1, 2):
            gates_unit.append(ZZ(wires=[j, j + 1], value=parameters[0, l]))
        if boundary_condition == 'closed':
            gates_unit.append(ZZ(wires=[N - 1, 0], value=parameters[0, l]))
        for i in range(N):
            gates_unit.append(RX(wires=[i, ], value=parameters[1, l]))
    qc = QuantumCircuit(nqubits=N, gates=gates_unit, device='GPU')

    return qc, projective_circuits


def xy_1d_hva_alt_circuit(N: int, depth: int, initial_parameters, boundary_condition):
    verify_circuit_input(N, depth, initial_parameters, (5, depth))

    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))
    gates = []

    # gates.extend([Hadamard(wires=[i, ]) for i in range(N)])
    for l in range(depth):
        for j in range(0, N - 1, 2):
            gates.append(ZZ(wires=[j, j + 1], value=parameters[0, l]))
            gates.append(YY(wires=[j, j + 1], value=parameters[1, l]))

        for j in range(1, N - 1, 2):
            gates.append(ZZ(wires=[j, j + 1], value=parameters[2, l]))
            gates.append(YY(wires=[j, j + 1], value=parameters[3, l]))
        if boundary_condition == 'closed':
            gates.append(ZZ(wires=[N - 1, 0], value=parameters[2, l]))
            gates.append(YY(wires=[N - 1, 0], value=parameters[3, l]))

        gates.extend([RX(wires=[i, ], value=parameters[4, l]) for i in range(N)])
    return gates


def xy_1d_hva_circuit(N: int, depth: int, initial_parameters, boundary_condition):
    verify_circuit_input(N, depth, initial_parameters, (3, depth))

    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))
    gates = []

    # gates.extend([Hadamard(wires=[i, ]) for i in range(N)])
    for l in range(depth):
        for j in range(0, N - 1, 2):
            gates.append(ZZ(wires=[j, j + 1], value=parameters[0, l]))
            gates.append(ZZ(wires=[j, j + 1], value=parameters[0, l]))

        if boundary_condition == 'closed':
            gates.append(ZZ(wires=[N - 1, 0], value=parameters[0, l]))

        for j in range(0, N - 1, 2):
            gates.append(YY(wires=[j, j + 1], value=parameters[1, l]))
            gates.append(YY(wires=[j, j + 1], value=parameters[1, l]))

        if boundary_condition == 'closed':
            gates.append(YY(wires=[N - 1, 0], value=parameters[1, l]))

        gates.extend([RX(wires=[i, ], value=parameters[2, l]) for i in range(N)])
    return gates


def tfi_2d_hva_circuit(N: int, depth: int, edge_coloring: dict, initial_parameters, f_or_af: 'f'):
    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))
    gates = []
    print(list(edge_coloring.values()))
    if f_or_af == 'f':
        gates.extend([Hadamard(wires=[i]) for i in range(N)])
    elif f_or_af == 'af':
        for edge in list(edge_coloring.values())[-1]:
            gates.append(PauliX(wires=[edge[0]]))

    for d in range(depth):
        for alpha, graph in enumerate(edge_coloring.values()):
            print([edge for edge in graph])
            gates.extend([ZZ(wires=edge, value=parameters[0, d]) for edge in graph])
        gates.extend([RX(wires=[i], value=parameters[1, d]) for i in range(N)])

    return gates


def xxz_1d_hva_circuit(N: int, depth: int, initial_parameters, boundary_condition):
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
        if boundary_condition == 'closed':
            gates.append(ZZ(wires=[N - 1, 0], value=parameters[0, 0, l]))
            gates.append(YY(wires=[N - 1, 0], value=parameters[0, 1, l]))
            gates.append(XX(wires=[N - 1, 0], value=parameters[0, 1, l]))

        for j in range(0, N - 1, 2):
            gates.append(ZZ(wires=[j, j + 1], value=parameters[1, 0, l]))
            gates.append(YY(wires=[j, j + 1], value=parameters[1, 1, l]))
            gates.append(XX(wires=[j, j + 1], value=parameters[1, 1, l]))

    return gates


def xxz_1d_projective_hva_circuit(N: int, depth: int, projectors, initial_parameters, boundary_condition: str, nmeas:int):
    verify_circuit_input(N, depth, initial_parameters, (2, 2, depth))

    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))

    projector_combinations = list(it.product((0, 1), repeat=nmeas))
    projective_circuits = []
    gpus = get_available_devices('GPU')

    for num, comb in enumerate(projector_combinations):
        gpu_num = int(num % len(gpus))
        projectors_comb = [projectors[i] for i in comb]
        gates_proj = []
        meas_count = 0
        for i in range(0, N - 1, 2):
            gates_proj.append(PauliX(wires=[i]))
            gates_proj.append(PauliX(wires=[i + 1]))
            gates_proj.append(Hadamard(wires=[i]))
            gates_proj.append(CNOT(wires=[i, i + 1]))

        for l in range(depth):
            for j in range(1, N - 1, 2):
                gates_proj.append(ZZ(wires=[j, j + 1], value=parameters[0, 0, l]))
                gates_proj.append(YY(wires=[j, j + 1], value=parameters[0, 1, l]))
                gates_proj.append(XX(wires=[j, j + 1], value=parameters[0, 1, l]))

            if boundary_condition == 'closed':
                gates_proj.append(ZZ(wires=[N - 1, 0], value=parameters[0, 0, l]))
                gates_proj.append(YY(wires=[N - 1, 0], value=parameters[0, 1, l]))
                gates_proj.append(XX(wires=[N - 1, 0], value=parameters[0, 1, l]))

            for j in range(0, N - 1, 2):
                gates_proj.append(ZZ(wires=[j, j + 1], value=parameters[1, 0, l]))
                gates_proj.append(YY(wires=[j, j + 1], value=parameters[1, 1, l]))
                gates_proj.append(XX(wires=[j, j + 1], value=parameters[1, 1, l]))
            if meas_count<nmeas:
                gates_proj.append(Projector(pi=projectors_comb[meas_count], wires=[l, ]))
                meas_count+=1
        projective_circuits.append([QuantumCircuit(nqubits=N, gates=gates_proj, device_number=gpu_num, device='GPU'), ])
        projective_circuits[num][0].execute()
        projective_circuits[num].append(tf.math.real(
            tf.reduce_prod(tf.stack([l.Z for l in projective_circuits[num][0].circuit.layers if l._is_projector]))) ** 2
                                        )

    gates_unit = []

    for i in range(0, N - 1, 2):
        gates_unit.append(PauliX(wires=[i]))
        gates_unit.append(PauliX(wires=[i + 1]))
        gates_unit.append(Hadamard(wires=[i]))
        gates_unit.append(CNOT(wires=[i, i + 1]))

    for l in range(depth):
        for j in range(1, N - 1, 2):
            gates_unit.append(ZZ(wires=[j, j + 1], value=parameters[0, 0, l]))
            gates_unit.append(YY(wires=[j, j + 1], value=parameters[0, 1, l]))
            gates_unit.append(XX(wires=[j, j + 1], value=parameters[0, 1, l]))

        if boundary_condition == 'closed':
            gates_unit.append(ZZ(wires=[N - 1, 0], value=parameters[0, 0, l]))
            gates_unit.append(YY(wires=[N - 1, 0], value=parameters[0, 1, l]))
            gates_unit.append(XX(wires=[N - 1, 0], value=parameters[0, 1, l]))

        for j in range(0, N - 1, 2):
            gates_unit.append(ZZ(wires=[j, j + 1], value=parameters[1, 0, l]))
            gates_unit.append(YY(wires=[j, j + 1], value=parameters[1, 1, l]))
            gates_unit.append(XX(wires=[j, j + 1], value=parameters[1, 1, l]))
    qc = QuantumCircuit(nqubits=N, gates=gates_unit, device='GPU')

    return qc, projective_circuits


def xxz_1d_projective_hva_circuit_multi_measure(N: int, depth: int, projectors, initial_parameters,
                                                boundary_condition: str, nmeas:int):
    verify_circuit_input(N, depth, initial_parameters, (2, 2, depth))

    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))

    projective_circuits = []

    # comb = np.random.randint(0,2, nmeas)
    comb = [0]*nmeas
    print(f'Projectors applied: {comb}')
    projectors_comb = [projectors[i] for i in comb]
    gates_proj = []
    meas_count = 0
    for i in range(0, N - 1, 2):
        gates_proj.append(PauliX(wires=[i]))
        gates_proj.append(PauliX(wires=[i + 1]))
        gates_proj.append(Hadamard(wires=[i]))
        gates_proj.append(CNOT(wires=[i, i + 1]))

    for l in range(depth):
        for j in range(1, N - 1, 2):
            gates_proj.append(ZZ(wires=[j, j + 1], value=parameters[0, 0, l]))
            gates_proj.append(YY(wires=[j, j + 1], value=parameters[0, 1, l]))
            gates_proj.append(XX(wires=[j, j + 1], value=parameters[0, 1, l]))

        if boundary_condition == 'closed':
            gates_proj.append(ZZ(wires=[N - 1, 0], value=parameters[0, 0, l]))
            gates_proj.append(YY(wires=[N - 1, 0], value=parameters[0, 1, l]))
            gates_proj.append(XX(wires=[N - 1, 0], value=parameters[0, 1, l]))

        if meas_count<nmeas:
            locs = np.random.choice(list(range(N)), min(nmeas-meas_count, N//2))
            print(locs, f'depth {l}')
            for loc in locs:
                gates_proj.append(Projector(pi=projectors_comb[meas_count], wires=[loc, ]))
                meas_count+=1

        for j in range(0, N - 1, 2):
            gates_proj.append(ZZ(wires=[j, j + 1], value=parameters[1, 0, l]))
            gates_proj.append(YY(wires=[j, j + 1], value=parameters[1, 1, l]))
            gates_proj.append(XX(wires=[j, j + 1], value=parameters[1, 1, l]))
        if meas_count < nmeas:
            locs = np.random.choice(list(range(N)), min(nmeas-meas_count, N//2))
            print(locs, f'depth {l}')
            for loc in locs:
                gates_proj.append(Projector(pi=projectors_comb[meas_count], wires=[loc, ]))
                meas_count += 1
    projective_circuits.append([QuantumCircuit(nqubits=N, gates=gates_proj, device='GPU'), ])
    projective_circuits[0][0].execute()
    projective_circuits[0].append(tf.constant(1.0, TF_FLOAT_DTYPE))

    gates_unit = []

    for i in range(0, N - 1, 2):
        gates_unit.append(PauliX(wires=[i]))
        gates_unit.append(PauliX(wires=[i + 1]))
        gates_unit.append(Hadamard(wires=[i]))
        gates_unit.append(CNOT(wires=[i, i + 1]))

    for l in range(depth):
        for j in range(1, N - 1, 2):
            gates_unit.append(ZZ(wires=[j, j + 1], value=parameters[0, 0, l]))
            gates_unit.append(YY(wires=[j, j + 1], value=parameters[0, 1, l]))
            gates_unit.append(XX(wires=[j, j + 1], value=parameters[0, 1, l]))

        if boundary_condition == 'closed':
            gates_unit.append(ZZ(wires=[N - 1, 0], value=parameters[0, 0, l]))
            gates_unit.append(YY(wires=[N - 1, 0], value=parameters[0, 1, l]))
            gates_unit.append(XX(wires=[N - 1, 0], value=parameters[0, 1, l]))

        for j in range(0, N - 1, 2):
            gates_unit.append(ZZ(wires=[j, j + 1], value=parameters[1, 0, l]))
            gates_unit.append(YY(wires=[j, j + 1], value=parameters[1, 1, l]))
            gates_unit.append(XX(wires=[j, j + 1], value=parameters[1, 1, l]))
    qc = QuantumCircuit(nqubits=N, gates=gates_unit, device='GPU')

    return qc, projective_circuits


def xxz_1d_projective_hva_circuit_single_measure(N: int, depth: int, projectors, initial_parameters,
                                                boundary_condition: str, nmeas:int):
    verify_circuit_input(N, depth, initial_parameters, (2, 2, depth))

    parameters = tf.Variable(initial_parameters,
                             constraint=lambda x: tf.clip_by_value(x, 0, 2 * np.pi))

    projective_circuits = []

    comb = [0]*nmeas * depth
    print(f'Projectors applied: {comb}')
    projectors_comb = [projectors[i] for i in comb]
    gates_proj = []
    meas_count = 0
    for i in range(0, N - 1, 2):
        gates_proj.append(PauliX(wires=[i]))
        gates_proj.append(PauliX(wires=[i + 1]))
        gates_proj.append(Hadamard(wires=[i]))
        gates_proj.append(CNOT(wires=[i, i + 1]))

    for l in range(depth):
        for j in range(1, N - 1, 2):
            gates_proj.append(ZZ(wires=[j, j + 1], value=parameters[0, 0, l]))
            gates_proj.append(YY(wires=[j, j + 1], value=parameters[0, 1, l]))
            gates_proj.append(XX(wires=[j, j + 1], value=parameters[0, 1, l]))

        if boundary_condition == 'closed':
            gates_proj.append(ZZ(wires=[N - 1, 0], value=parameters[0, 0, l]))
            gates_proj.append(YY(wires=[N - 1, 0], value=parameters[0, 1, l]))
            gates_proj.append(XX(wires=[N - 1, 0], value=parameters[0, 1, l]))

        for j in range(0, N - 1, 2):
            gates_proj.append(ZZ(wires=[j, j + 1], value=parameters[1, 0, l]))
            gates_proj.append(YY(wires=[j, j + 1], value=parameters[1, 1, l]))
            gates_proj.append(XX(wires=[j, j + 1], value=parameters[1, 1, l]))
        locs = list(range(nmeas))
        for loc in locs:
            gates_proj.append(Projector(pi=projectors_comb[meas_count], wires=[loc, ]))
            meas_count += 1

    projective_circuits.append([QuantumCircuit(nqubits=N, gates=gates_proj, device='GPU'), ])
    projective_circuits[0][0].execute()
    projective_circuits[0].append(tf.constant(1.0, TF_FLOAT_DTYPE))

    gates_unit = []

    for i in range(0, N - 1, 2):
        gates_unit.append(PauliX(wires=[i]))
        gates_unit.append(PauliX(wires=[i + 1]))
        gates_unit.append(Hadamard(wires=[i]))
        gates_unit.append(CNOT(wires=[i, i + 1]))

    for l in range(depth):
        for j in range(1, N - 1, 2):
            gates_unit.append(ZZ(wires=[j, j + 1], value=parameters[0, 0, l]))
            gates_unit.append(YY(wires=[j, j + 1], value=parameters[0, 1, l]))
            gates_unit.append(XX(wires=[j, j + 1], value=parameters[0, 1, l]))

        if boundary_condition == 'closed':
            gates_unit.append(ZZ(wires=[N - 1, 0], value=parameters[0, 0, l]))
            gates_unit.append(YY(wires=[N - 1, 0], value=parameters[0, 1, l]))
            gates_unit.append(XX(wires=[N - 1, 0], value=parameters[0, 1, l]))

        for j in range(0, N - 1, 2):
            gates_unit.append(ZZ(wires=[j, j + 1], value=parameters[1, 0, l]))
            gates_unit.append(YY(wires=[j, j + 1], value=parameters[1, 1, l]))
            gates_unit.append(XX(wires=[j, j + 1], value=parameters[1, 1, l]))
    qc = QuantumCircuit(nqubits=N, gates=gates_unit, device='GPU')

    return qc, projective_circuits


def xxz_1d_perm_hva_circuit(N: int, depth: int, initial_parameters, boundary_condition, permutation):
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
        for layer_number in range(4):
            if permutation[l * 4 + layer_number] == 2:
                for j in range(1, N - 1, 2):
                    gates.append(ZZ(wires=[j, j + 1], value=parameters[0, 0, l]))
                if boundary_condition == 'closed':
                    gates.append(ZZ(wires=[N - 1, 0], value=parameters[0, 0, l]))

            elif permutation[l * 4 + layer_number] == 3:
                for j in range(1, N - 1, 2):
                    gates.append(YY(wires=[j, j + 1], value=parameters[0, 1, l]))
                    gates.append(XX(wires=[j, j + 1], value=parameters[0, 1, l]))
                if boundary_condition == 'closed':
                    gates.append(YY(wires=[N - 1, 0], value=parameters[0, 1, l]))
                    gates.append(XX(wires=[N - 1, 0], value=parameters[0, 1, l]))

            elif permutation[l * 4 + layer_number] == 0:
                for j in range(0, N - 1, 2):
                    gates.append(ZZ(wires=[j, j + 1], value=parameters[1, 0, l]))

            elif permutation[l * 4 + layer_number] == 1:
                for j in range(0, N - 1, 2):
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
    # for edge in edge_coloring['zz']:
    # Bellstates
    # gates.append(PauliX(wires=[edge[0]]))
    # gates.append(PauliX(wires=[edge[1]]))
    # gates.append(Hadamard(wires=[edge[0]]))
    # gates.append(CNOT(wires=[edge[0], edge[1]]))

    parameters = tf.Variable(initial_parameters,
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


def kitaev_ladder_circuit(depth: int, edge_coloring: dict, initial_parameters: np.ndarray):
    gates = []
    # max_site = max(max(i,j) for i,j in edge_coloring['zz'])
    # for edge in edge_coloring['xx']:
    #     print(edge)
    #     gates.append(PauliX(wires=[edge[0]]))
    #     gates.append(Hadamard(wires=[edge[0]]))
    #     gates.append(CNOT(wires=[edge[0], edge[1]]))
    for edge in edge_coloring['yy']:
        gates.append(PauliX(wires=[edge[0]]))
        print(edge)
        # gates.append(Hadamard(wires=[edge[0]]))
        # gates.append(Hadamard(wires=[edge[0]]))
        gates.append(CNOT(wires=[edge[0], edge[1]]))
    parameters = tf.Variable(initial_parameters,
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
