import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
import matplotlib.lines as mlines
from matplotlib import rcParams
from typing import List
import numpy as np


def _add_single_block(x, y, name, ax):
    """
    Add a single block of two gates.

    Args:
        *x (float)*:
            Layer position.

        *y (float)*:
            Qubit 1 position.

        *name (str)*:
            Gate name.

        *ax (pyplot.Axes)*:
            Axes for the figure.

    Returns (inplace):
        None

    """
    ax.add_patch(patches.Rectangle((x, y), 0.5, 0.5, facecolor='white', edgecolor='black'))
    tp = TextPath((x, y + 0.16), name, size=0.25)
    center = (tp.get_extents().xmax - tp.get_extents().xmin) / 2
    tp = TextPath((x + 0.24 - center, y + 0.16), name, size=0.25)
    ax.add_patch(patches.PathPatch(tp, color="black"))


def _add_control(x, y, ax):
    """
    Add a control circle for one gate.

    Args:
        *x (float)*:
            Layer position.

        *y (float)*:
            Qubit 1 position.

        *ax (pyplot.Axes)*:
            Axes for the figure.

    Returns (inplace):
        None

    """
    ax.add_patch(patches.Circle((x + 0.25, y + 0.25), 0.12, facecolor='white', edgecolor='black'))
    ax.add_line(mlines.Line2D([x + 0.165, x + 0.335], [y + 0.165, y + 0.335], zorder=3, color='black', linewidth=1))
    ax.add_line(mlines.Line2D([x + 0.165, x + 0.335], [y + 0.335, y + 0.165], zorder=3, color='black', linewidth=1))

def _add_target(x, y, ax):
    """
    Add a target circle for one gate.

    Args:
        *x (float)*:
            Layer position.

        *y (float)*:
            Qubit 1 position.

        *ax (pyplot.Axes)*:
            Axes for the figure.

    Returns (inplace):
        None

    """
    ax.add_patch(patches.Circle((x + 0.25, y + 0.25), 0.076, facecolor='black', edgecolor='black'))



def _add_double_block(x, y, name, ax):
    """
    Add a double block of two gates.

    Args:
        *x (float)*:
            Layer position.

        *y (float)*:
            Qubit 1 position.

        *name (str)*:
            Gate name.

        *ax (pyplot.Axes)*:
            Axes for the figure.

    Returns (inplace):
        None

    """
    ax.add_patch(patches.Rectangle((x, y), 0.5, 1.5, facecolor='white', edgecolor='black'))
    tp = TextPath((x, y + 0.66), name, size=0.25)
    center = (tp.get_extents().xmax - tp.get_extents().xmin) / 2
    tp = TextPath((x + 0.24 - center, y + 0.66), name, size=0.25)
    ax.add_patch(patches.PathPatch(tp, color="black"))


def _add_connector(x, y1, y2, ax):
    """
    Add a connector line between two gates.

    Args:
        *x (float)*:
            Layer position.

        *y1 (float)*:
            Gate 1 position.

        *y2 (float)*:
            Gate 2 position.

        *ax (pyplot.Axes)*:
            Axes for the figure.

    Returns (inplace):
        None

    """
    ax.add_line(mlines.Line2D([x + 0.25, x + 0.25], [y1 + 0.25, y2], zorder=0.5, color='black', linewidth=2))


def _resolve_layout(N: int, gate_wires: list):
    """
    Resolve the layout per layer to make sure the space usage is optimal

    Args:
        *N (int)*:
            The number of qubits.

        *gate_wires (list)*:
            List of lists containing the gate wires.


    Returns (dict, int):
        First return value is dictionary containing the shift with respect to zero for each gate in the layer.
        The second return value is an integer specifying the maximum shift necessary.

    """
    shift = dict(zip(range(N), [0 for _ in range(N)]))
    for gw in gate_wires:
        if len(gw) == 2:
            max_shift_gw = max([shift[gw[0]], shift[gw[1]]])
            shift[gw[0]], shift[gw[1]] = (max_shift_gw, max_shift_gw)
            mingw, maxgw = (min(gw), max(gw))
            if maxgw - mingw > 1:
                for w in range(mingw + 1, maxgw):
                    if shift[w] == max_shift_gw:
                        shift[w] += 1
        if len(gw) > 2:
            raise NotImplementedError('Drawing circuits is not implemented for gates working on more than 2 qubits.')
    return shift, max(shift.values())


def draw_circuit(layers: List[list], N: int):
    """
    Draw the circuit from a list of lists containing the gate names with format:
    <NAME>_<W1>_<W2>_...

    For now this only works for gates working on 1 and 2 qubits.

    Args:
        *layers (list)*:
            List of lists containing strings corresponding to the gates.

        *N (int)*:
            The number of qubits.

    Returns (inplace):
        None

    """
    line_shift = 0.35
    total_shift = 0
    inverse_map = dict(zip(range(N), reversed(range(N))))
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    rcParams['font.family'] = 'georgia'
    L = len(layers)
    plt.autoscale(False)
    for l in range(L):
        gate_name = []
        gate_wires = []
        for gate in layers[l]:
            split_gate = gate.split('_')

            gate_name.append(split_gate[0])
            gate_wires.append(np.sort([inverse_map[int(i)] for i in split_gate[1:] if i != '_']))
        layout_shifts, max_shift = _resolve_layout(N, gate_wires)
        for (gw, gn) in zip(gate_wires, gate_name):
            if len(gw) == 1:
                _add_single_block(l + total_shift, gw[0], gn, ax)
            if len(gw) == 2:
                if gn == 'CNOT':
                    _add_control(l + total_shift + layout_shifts[gw[0]] * line_shift, gw[0], ax)
                    _add_target(l + total_shift + layout_shifts[gw[0]] * line_shift, gw[1], ax)
                    _add_connector(l + total_shift + layout_shifts[gw[0]] * line_shift, gw[0], gw[1], ax)
                elif gn in ['CZ', 'CRX', 'CRY', 'CRZ']:
                    _add_control(l + total_shift + layout_shifts[gw[0]] * line_shift, gw[0], ax)
                    _add_single_block(l + total_shift + layout_shifts[gw[1]] * line_shift, gw[1], gn[1:], ax)
                    _add_connector(l + total_shift + layout_shifts[gw[0]] * line_shift, gw[0], gw[1], ax)
                else:
                    if gw[0] == (gw[1] - 1):
                        _add_double_block(l + total_shift + layout_shifts[gw[0]] * line_shift, gw[0], gn, ax)
                    else:
                        for w in gw:
                            _add_single_block(l + total_shift + layout_shifts[w] * line_shift, w, gn, ax)
                        _add_connector(l + total_shift + layout_shifts[gw[0]] * line_shift, gw[0], gw[1], ax)
        total_shift += max_shift * 0.35
    for i in range(N):
        ax.add_line(mlines.Line2D([-0.4, L - 0.1 + total_shift], [i + 0.25, i + 0.25],
                                  zorder=0.5, color=[0.2, 0.2, 0.2], linewidth=5))

    ax.set_xlim([-0.5, L + total_shift])
    ax.set_ylim([-0.5, N])
    ax.axis('off')
    fig.set_size_inches(L + 0.5 + total_shift, N + 0.5)
    plt.show()
