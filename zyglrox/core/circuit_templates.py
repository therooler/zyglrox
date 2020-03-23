from zyglrox.core.gates import *


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
