# import pytest
from zyglrox.core.hamiltonians import Hamiltonian, TFI, HeisenbergXXZ
from zyglrox.core.topologies import *
import matplotlib.pyplot as plt


def kagome():
    model_graph = graph_kagome_12()

    inter = {'zz': model_graph, 'x': dict(zip(range(12), [[[i, ]] for i in range(12)]))}
    h = Hamiltonian(topology=model_graph, interactions=inter, model_parameters={'zz': 1, 'x': 4})
    ts = h.get_hamiltonian_terms()
    obs = h.get_observables()
    h.plot_lattice()
    h.get_hamiltonian()


def standard_topologies():
    h = TFI(topology='line', g=1.5, boundary_conditions='closed', L=8)
    # h.plot_lattice()
    h.plot_color_lattice()
    h.get_hamiltonian()


def site_skip():
    model_graph = {0: [[0, 1], [0, 1]], 2: [[1, 0]]}

    inter = {'zz': model_graph, }
    h = Hamiltonian(topology=model_graph, interactions=inter, model_parameters={'zz': 1, })


def kagome_torus_gs():
    model_graph = graph_kagome_12_torus()
    inter = {'zz': model_graph}
    h = Hamiltonian(topology=model_graph, interactions=inter, model_parameters={'zz': 1}, name='Kagome_torus_zz', k=50)
    # h.plot_lattice()
    h.get_hamiltonian()


def kagome_torus_gs_2():
    model_graph = graph_kagome_12_torus()

    inter = {'xx': model_graph, 'yy': model_graph, }
    h = Hamiltonian(topology=model_graph, interactions=inter, model_parameters={'xx': 1, 'yy': 1},
                    name='Kagome_torus_xxx', k=50)

    h.plot_color_lattice()
    h.get_hamiltonian()


def kagome_star_xxy():
    model_graph = {0: [[0, 11], [0, 1]], 1: [[1, 0], [1, 11], [1, 2], [1, 3]],
                   2: [[2, 1], [2, 3]], 3: [[3, 1], [3, 2], [3, 4], [3, 5]],
                   4: [[4, 3], [4, 5]], 5: [[5, 3], [5, 4], [5, 6], [5, 7]],
                   6: [[6, 5], [6, 7]], 7: [[7, 5], [7, 6], [7, 8], [7, 9]],
                   8: [[8, 7], [8, 9]], 9: [[9, 7], [9, 8], [9, 10], [9, 11]],
                   10: [[10, 9], [10, 11]], 11: [[11, 9], [11, 10], [11, 0], [11, 1]]}

    h = HeisenbergXXZ(topology=model_graph, delta=-1.0, name='Kagome_torus_xxy_12', k=50, )
    h.get_hamiltonian()
    plt.hist(h.groundstates.real)
    plt.show()
    # h.plot_lattice()
    # h.plot_color_lattice()
    h.get_hamiltonian()


def kagome_18b():
    model_graph = graph_kagome_18b()

    h = HeisenbergXXZ(topology=model_graph, delta=1.0, name='Kagome_torus_xxy_18', k=50, )
    h.plot_lattice()
    h.plot_color_lattice()
    plt.show()


def kagome_18b_torus():
    model_graph = graph_kagome_18b_torus()

    h = HeisenbergXXZ(topology=model_graph, delta=1.0, name='Kagome_torus_xxy_18', k=50, )
    h.plot_lattice()
    h.plot_color_lattice()
    plt.show()


def kagome_27():
    model_graph = graph_kagome_27()

    h = HeisenbergXXZ(topology=model_graph, delta=1.0, name='Kagome_torus_xxy_18', k=50)
    positions = graph_kagome_27()
    h.plot_lattice(pos=positions)
    h.plot_color_lattice(pos=positions)
    plt.show()
    #
    #
def kagome_27_torus():
    model_graph = {0: [[0, 1], [0, 3], [0, 16], [0, 17]], 1: [[1, 0], [1, 2], [1, 4], [1, 17]],
                   2: [[2, 1], [2, 3], [2, 4], [2, 5]], 3: [[3, 2], [3, 5], [3, 0], [3, 16]],
                   4: [[4, 1], [4, 2], [4, 6], [4, 7]], 5: [[5, 3], [5, 2], [5, 9], [5, 6]],
                   6: [[6, 4], [6, 7], [6, 5], [6, 9]], 7: [[7, 4], [7, 6], [7, 8], [7, 11]],
                   8: [[8, 7], [8, 9], [8, 10], [8, 11]], 9: [[9, 5], [9, 8], [9, 10], [9, 6]],
                   10: [[10, 9], [10, 8], [10, 15], [10, 12]], 11: [[11, 7], [11, 8], [11, 12], [11, 13]],
                   12: [[12, 11], [12, 13], [12, 10], [12, 15]], 13: [[13, 11], [13, 12], [13, 14], [13, 17]],
                   14: [[14, 13], [14, 15], [14, 16], [14, 17]], 15: [[15, 10], [15, 14], [15, 16], [15, 12]],
                   16: [[16, 14], [16, 15], [16, 0], [16, 3]], 17: [[17, 14], [17, 13], [17, 1], [17, 0]]}

    h = HeisenbergXXZ(topology=model_graph, delta=1.0, name='Kagome_torus_xxy_18', k=50, )
    h.plot_lattice()
    h.plot_color_lattice()
    plt.show()


if __name__ == "__main__":
    # kagome()
    # standard_topologies()
    # site_skip()
    # kagome_18b()
    # kagome_18b_torus()
    kagome_27()
