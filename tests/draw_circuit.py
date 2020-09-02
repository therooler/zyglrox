import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.textpath import TextPath
import matplotlib.lines as mlines
from matplotlib import rcParams

layer1 = 'H_0,H_1,H_2,H_3,H_4,H_5,H_6,H_7'.split(',')
layer2 = 'CNOT_1_0,ZZ_2_3,CNOT_4_5,ZZ_6_7'.split(',')
layer3 = 'CZ_0_2,ZZ_1_3,CRX_4_7,ZZ_5_6'.split(',')
layers = [layer1, layer2, layer3, layer1]
L = len(layers)
N = 8
fig, ax = plt.subplots()
ax.set_aspect('equal')
rcParams['font.family'] = 'georgia'

plt.autoscale(False)


def _add_control(x, y, ax):
    ax.add_patch(patches.Circle((x+0.25, y+0.25), 0.076, facecolor='black', edgecolor='black'))


def _add_target(x, y, ax):
    ax.add_patch(patches.Circle((x + 0.25, y + 0.25), 0.12, facecolor='white', edgecolor='black'))
    ax.add_line(mlines.Line2D([x+0.165, x + 0.335], [y+0.165 , y + 0.335], zorder=3, color='black', linewidth=1))
    ax.add_line(mlines.Line2D([x+0.165, x + 0.335], [y + 0.335, y +0.165], zorder=3, color='black', linewidth=1))

def _add_single_block(x, y, name, ax):
    ax.add_patch(patches.Rectangle((x, y), 0.5, 0.5, facecolor='white', edgecolor='black'))
    tp = TextPath((x, y + 0.16), name, size=0.25)
    center = (tp.get_extents().xmax - tp.get_extents().xmin) / 2
    tp = TextPath((x + 0.24 - center, y + 0.16), name, size=0.25)
    ax.add_patch(patches.PathPatch(tp, color="black"))


def _add_double_block(x, y, name, ax):
    ax.add_patch(patches.Rectangle((x, y), 0.5, 1.5, facecolor='white', edgecolor='black'))
    tp = TextPath((x, y + 0.66), name, size=0.25)
    center = (tp.get_extents().xmax - tp.get_extents().xmin) / 2
    tp = TextPath((x + 0.24 - center, y + 0.66), name, size=0.25)
    ax.add_patch(patches.PathPatch(tp, color="black"))


def _add_connector(x, y1, y2, ax):
    ax.add_line(mlines.Line2D([x + 0.25, x + 0.25], [y1+0.25, y2+0.25], zorder=0.5, color='black', linewidth=2))


def _resolve_layout(N, gate_wires):
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


line_shift = 0.35
total_shift = 0
inverse_map = dict(zip(range(N), reversed(range(N))))

for l in range(L):
    gate_name = []
    gate_wires = []
    for gate in layers[l]:
        split_gate = gate.split('_')

        gate_name.append(split_gate[0])
        gate_wires.append([inverse_map[int(i)] for i in split_gate[1:] if i != '_'])
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
                    _add_connector(l + total_shift + layout_shifts[w] * line_shift, gw[0], gw[1], ax)
    total_shift += max_shift * 0.35
for i in range(N):
    ax.add_line(mlines.Line2D([-0.4, L - 0.1 + total_shift], [i + 0.25, i + 0.25],
                              zorder=0.5, color=[0.2, 0.2, 0.2], linewidth=5))

ax.set_xlim([-0.5, L + total_shift])
ax.set_ylim([-0.5, N])
ax.axis('off')
fig.set_size_inches(L + 0.5 + total_shift, N + 0.5)

plt.show()
