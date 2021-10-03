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


from zyglrox.core.observables import Observable
from zyglrox.core.utils import tf_kron
from zyglrox.core.edge_coloring import applyHeuristic
from zyglrox.core.topologies import *
from zyglrox.core._config import TF_COMPLEX_DTYPE

import os
from operator import itemgetter
from typing import List, Union

import scipy.sparse.linalg as ssla
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import networkx as nx
import tensorflow as tf


class Hamiltonian(object):
    paulis = {'I': scipy.sparse.csr_matrix(np.eye(2).astype(np.complex64)),
              'x': scipy.sparse.csr_matrix(np.array([[0, 1], [1, 0]]).astype(np.complex64)),
              'y': scipy.sparse.csr_matrix(np.array([[0, -1j], [1j, 0]]).astype(np.complex64)),
              'z': scipy.sparse.csr_matrix(np.array([[1, 0], [0, -1]]).astype(np.complex64))}

    def __init__(self, topology: dict, interactions: dict, model_parameters: dict = {}, **kwargs):
        r"""
        Hamiltonian is the abstract class for defining Hamiltonians of physical systems. For our purposes,
        the Hamiltonian exists of three components:

        1. A topology :math:`\Lambda` defining the lattice that our model lives on. This can be as simple as a line or
        square lattice, or as complicated as a fully connected model where each site is physically connected to each other site.

        2. An interaction graph :math:`\Lambda_a \subseteq \Lambda` which is sub-graph of the full topology with a corresponding
        string :math:`\alpha\beta\ldots` with :math:`\alpha,\beta,\ldots \in \{x,y,z\}` that indicates which Pauli interaction we are considering.

        3. Model parameters that correspond to the strength of the interactions. This can be either a single value, so that the interaction
        strength is the same everywhere, or this can be a set of nodes where each vertex has its own interaction strength.

        With these three ingredients, a wide range of spin models can be described. When the this class is initialized,
        a subfolder ``./hamiltonians`` is automatically created relative to the root. Additionally, one can pass the ``file_path`` kwarg
        to specify a different location.

        Args:
            *topology (dict)*:
                A dict with nodes as keys and a list of edges as values.

            *interactions (dict)*:
                A dict with strings of the type :math:`\alpha\beta\ldots` as keys and topology dicts as values.

            *model_parameters (dict)*:
                A dict with strings of the type :math:`\alpha\beta\ldots` as keys and floats as values. If the interaction
                strength varies per site this can be a dict of vertices with each its own interaction strength

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """

        ## MODEL TOPOLOGY ##
        assert isinstance(topology, dict), "topology must be a dictionary, received {}".format(
            type(topology))
        assert all(isinstance(x, int) for x in topology.keys()), "All keys of the topology dict must be integers"
        assert set(topology.keys()) == set(range(max(
            topology.keys()) + 1)), "Topology must be a list of consecutive integers starting at 0, received {}".format(
            topology.keys())
        assert all(isinstance(x, list) for y in topology.values() for x in
                   y), "All values of the topology dict must be list of lists"
        assert all(
            len(x) == 2 for y in topology.values() for x in y), "All edges in the topology dict must be length 2"
        assert all(i == x[0] for i, y in topology.items() for x in
                   y), "All keys in the topology dict must be equal to the first index of the supplied edges"

        # self.nsites = max(topology.keys()) + 1
        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1

        ## MODEL INTERACTIONS ##
        assert isinstance(interactions, dict), "interactions must be a dictionary, received {}".format(
            type(interactions))
        assert all(isinstance(x, str) for x in interactions.keys()), "All keys of the interactions dict must be strings"
        for k, v in interactions.items():
            assert [x in ['x', 'y', 'z'] for x in
                    k], "Interaction must be composed of x, y, z interactions, received {}".format(k)
            assert isinstance(v, (dict,
                                  str)), "The values of the interactions must be either str or a dictionary, received {} for key '{}'".format(
                type(v), k)
            if isinstance(v, dict):
                assert all(isinstance(x, int) for x in v.keys()), \
                    "All keys of the interactions dict must be integers, received {} for key '{}'".format(v.keys(), k)
                assert all(len(set(x)) == len(x) for y in v.values() for x in y), \
                    "All vertices in the interaction must be unique, received {} for key '{}'".format(
                        v.values(), k)
                assert all(isinstance(x, list) for y in v.values() for x in y), \
                    "All values of the interactions dict must be list of lists, received {} for key '{}'".format(
                        list(v.values()), k)
                assert all(len(x) == len(k) for y in v.values() for x in y), \
                    "Interaction with name '{}' does not have edges with length {} ".format(k, len(k))
                assert all(i == x[0] for i, y in v.items() for x in y), \
                    "All keys in the interactions dict must be equal to the first index of the supplied edges"
                assert all(max(x) < self.nsites for y in v.values() for x in y), \
                    "Interaction '{}' has site interactions {}, that are not defined in the model topology with nsites = {}".format(
                        k, list(v.values()), self.nsites)
            if isinstance(v, str):
                if v == 'topology':
                    interactions[k] = topology
                else:
                    raise ValueError('only allowed string for interaction is "topology", received {}'.format(v))
        ## MODEL PARAMETERS ##
        assert all(x in interactions.keys() for x in model_parameters.keys()), \
            "model_parameters and interactions do not match: {} and {}".format(model_parameters.keys(),
                                                                               interactions.keys())
        assert all(isinstance(x, (int, float, dict)) for x in model_parameters.values()), \
            "received parameter that is not an int or float: {}".format(model_parameters.values())
        if all([isinstance(x, dict) for x in model_parameters.values()]):
            pass

        self.model_parameters = dict(zip(interactions.keys(), [1.0 for _ in range(len(interactions))]))
        for parameter, value in model_parameters.items():
            self.model_parameters[parameter] = value

        topology = remove_double_counting(topology)
        interactions = {k: remove_double_counting(v) for k, v in interactions.items()}

        self.topology = topology
        self.interactions = interactions
        self.filepath = kwargs.get('filepath', './hamiltonians')
        self.name = kwargs.pop('name', 'unnamed_model')
        self.H = None
        self.colored_edges = None
        self.k = kwargs.pop('k', self.nsites)
        self.interaction_order = sorted(self.interactions.keys())
        self.link_order = {}
        for term in self.interaction_order:
            self.link_order[term] = []
            for link in self.interactions[term].values():
                for l in link:
                    self.link_order[term].append(l)
        self.interaction_slices = {}
        if self.filepath[-1] != '/':
            self.filepath += '/'
            if self.name[-1] != '/':
                self.name += '/'
        if not os.path.exists(self.filepath):
            print('Path {} does not exist, so we create it now'.format(self.filepath))
            os.makedirs(self.filepath, exist_ok=True)

    def get_hamiltonian_terms(self) -> np.ndarray:
        """
        Get all the interactions in the Hamiltonian and add them to a 1d array. When calculating expectation values, this
        array can be used to multiply with the observables to get the energies.

        Returns (np.ndarray):
            Array with interaction strengths according to ``self.interaction_order``, a sorted list of the interactions provided.

        """
        w = []
        total_size = 0
        for term in self.interaction_order:
            beginning = total_size
            dict_size = sum(list(len(d) for d in self.interactions[term].values()))
            total_size += dict_size
            end = total_size
            self.interaction_slices[term] = (beginning, end)
            if isinstance(self.model_parameters[term], (int, float)):
                w.extend([self.model_parameters[term] for _ in range(dict_size)])
            else:
                for l in self.link_order[term]:
                    w.append(self.model_parameters[term][tuple(l)])
        return np.array(w)

    def get_observables(self) -> List[Observable]:
        r"""
        Get a list of ``Observable`` objects corresponding to all the terms in the hamiltonian.
        The order of the observables is according to ``self.interaction_order``, a sorted list of the interactions provided.

        Returns (list):
            A list of ``Observable`` objects.

        """

        observables = []
        for term in self.interaction_order:
            for link in self.interactions[term].values():
                for l in link:
                    if len(l) == 1:
                        observables.append(Observable(term, wires=l))
                    else:
                        p = tf.constant(Hamiltonian.paulis[term[0]].toarray(), dtype=TF_COMPLEX_DTYPE)
                        for i in range(1, len(l)):
                            s = tf.constant(Hamiltonian.paulis[term[i]].toarray(), dtype=TF_COMPLEX_DTYPE)
                            p = tf_kron(p, s)
                        observables.append(Observable('Hermitian', wires=l, value=p,
                                                      name=''.join([s for s in term])))
        return observables

    def get_hamiltonian(self):
        """
        Get a sparse matrix representation of the hamiltonian and calculate the eigenvalues and eigenvectors.
        When the system is degenerate, we store all :math:`N` degenerate eigenstates and energies. The Hamiltonian
        is automatically saved in the ``./hamiltonians`` path or in the otherwise specified ``file_path`` kwarg

        Returns (inplace):
            None

        """

        if not os.path.exists(self.filepath + self.name + 'energy.npy'):
            if self.H is None:
                self._build_hamiltonian()
            if not os.path.exists(self.filepath + self.name):
                os.mkdir(self.filepath + self.name)
            print("File " + self.filepath + self.name + "H.npz"
                  + " not found, creating and saving Hamiltonian...")
            scipy.sparse.save_npz(self.filepath + self.name + "H.npz", self.H)
            energies, eigenvectors = ssla.eigsh(self.H, k=self.k, which='SA')
            idx = np.argsort(energies)
            energies, eigenvectors = (energies[idx], eigenvectors[:, idx])
            self.d = self._find_degeneracy(energies)
            if self.d > 1:
                print("WARNING: Lanczos method is unstable for degenerate spectrum.")
            if self.d == self.k:
                raise ValueError(
                    "More than {} degenerate ground states found, increase scipy.sparse.linalg.eigsh k-value".format(
                        self.k))
            self.energies = energies[:self.d]
            self.groundstates = eigenvectors[:, :self.d].reshape((-1, self.d))
            self.gs = self.groundstates[:, 0]
            np.save(self.filepath + self.name + "energy", energies[:self.d])
            np.save(self.filepath + self.name + "groundstate", self.groundstates)
            print("Ground state energies = {}".format(self.energies))

        else:
            # load the matrix and eigenvectors
            self.H = scipy.sparse.load_npz(self.filepath + self.name + "H.npz")
            self.energies = np.load(self.filepath + self.name + "energy.npy")
            self.groundstates = np.load(self.filepath + self.name + "groundstate.npy")
            self.gs = self.groundstates[:, 0]

            print("Ground state energies = {}".format(self.energies[0]))

    @staticmethod
    def _find_degeneracy(E: np.ndarray) -> int:
        """
        Calculate the degeneracy based on the given energies.

        Args:
            *E (int)*:
                The energies of the system

        Returns (int):
            The number of degenerate states.

        """
        degeneracy = 0
        while np.allclose(E[degeneracy], E[degeneracy + 1]):
            degeneracy += 1
        if degeneracy > 0:
            print('The ground state is {}-fold degenerate'.format(degeneracy + 1))
        else:
            print('No degeneracy')
        return degeneracy + 1

    def _build_hamiltonian(self):
        """
        Build the matrix representation of the Hamiltonian based on the interactions and model parameters

        Returns (inplace):
            None

        """
        if self.nsites > 11:
            print("WARNING: nsites = {} with dim(H) = {}x{} so this may take a while".format(
                self.nsites, 2 ** self.nsites, 2 ** self.nsites)
            )

        # the hamiltonian has shape 2^N x 2^N
        self.H = scipy.sparse.csr_matrix((int(2 ** self.nsites),
                                          (int(2 ** self.nsites))), dtype=complex)

        for interaction, graph in self.interactions.items():
            for links in graph.values():
                for l in links:
                    if len(l) == 1:
                        tprod = ["I" for _ in range(self.nsites)]
                        tprod[l[0]] = interaction
                        p = Hamiltonian.paulis[tprod[0]]
                        # build the full tensorproduct recursively.
                        for op in range(1, self.nsites):
                            p = scipy.sparse.kron(p, Hamiltonian.paulis[tprod[op]], format='csr')
                        if isinstance(self.model_parameters[interaction], dict):
                            self.H += self.model_parameters[interaction][tuple(l)] * p
                        else:
                            self.H += self.model_parameters[interaction] * p
                    else:
                        tprod = ["I" for _ in range(self.nsites)]
                        for i, s in enumerate(l):
                            tprod[s] = interaction[i]
                        p = Hamiltonian.paulis[tprod[0]]
                        # build the full tensorproduct recursively.
                        for op in range(1, self.nsites):
                            p = scipy.sparse.kron(p, Hamiltonian.paulis[tprod[op]], format='csr')
                        if isinstance(self.model_parameters[interaction], dict):
                            self.H += self.model_parameters[interaction][tuple(l)] * p
                        else:
                            self.H += self.model_parameters[interaction] * p

    def draw_lattice(self, **kwargs):
        """
        Use Networkx to plot a Kamada-Kawai layout of the lattice. Takes the kwargs ``pos`` which is a dict of
        vertices and coordinates that indicates the location of the vertices in the plot.

        Args:
            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """
        positions = kwargs.pop('pos', None)
        g = self.get_nx_graph()
        labels = dict(zip(g.nodes(), [str(i + 1) for i in range(len(g.nodes()))]))
        if positions is not None:
            nx.drawing.draw_networkx(g, labels=labels, label_color='r', pos=positions)
        else:
            nx.drawing.draw_kamada_kawai(g, labels=labels, label_color='r')
        plt.show()

    def draw_color_lattice(self, **kwargs):
        r"""
        Use Networkx to plot an edge coloring of the graph. Makes use of ``applyHeuristic`` in ``zyglrox.core.edge_coloring``
        to find a suitable edge coloring. Per default uses the Kamada-Kawai layout of the lattice. 
        Takes the kwargs ``pos`` which is a dict of vertices and coordinates  that indicates the location of the vertices in the plot.
        
        Args:
            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """
        g = self.get_nx_graph()

        labels = dict(zip(g.nodes(), [str(i + 1) for i in range(len(g.nodes))]))
        positions = kwargs.pop('pos', None)
        self.get_colored_edges(g)

        if positions is not None:
            nx.drawing.draw_networkx_edges(g, positions, edge_list=g.edges(), edge_color=self.edge_coloring, width=8,
                                           alpha=0.5)
            nx.drawing.draw_networkx_nodes(g, positions, node_list=g.nodes(), node_color='white',
                                           edgecolors='black')
            nx.drawing.draw_networkx_labels(g, positions, labels=labels, label_color='r', font_size=10)
        else:
            nx.drawing.draw_networkx_edges(g, nx.kamada_kawai_layout(g), edge_list=g.edges(),
                                           edge_color=self.edge_coloring, width=8,
                                           alpha=0.5)
            nx.drawing.draw_networkx_nodes(g, nx.kamada_kawai_layout(g), node_list=g.nodes(), node_color='white',
                                           edgecolors='black')
            nx.drawing.draw_networkx_labels(g, nx.kamada_kawai_layout(g), labels=labels, label_color='r', font_size=10)
        plt.title("{} Site lattice".format(self.nsites))
        plt.show()

    def get_colored_edges(self, g):
        if self.colored_edges == None:
            max_degree = max([val for (_, val) in g.degree()])
            # assert max_degree < 5, NotImplementedError(
            #     "If the number of degrees is larger than 4, we need to add code to handle this")
            if not applyHeuristic(g, max_degree, 50, 50):
                print("Trying for degree {}+1".format(max_degree))
                applyHeuristic(g, max_degree + 1, 50, 50)
            self.edge_coloring = [g[e[0]][e[1]]['color'] for e in g.edges()]
            self.colored_edges = {}
            color_names = {0: 'purple', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'color5', 5: 'color6', 6: 'color7',
                           7: 'color8',
                           8: 'color9', 9: 'color10', 10: 'color11', 11: 'color12', 12: 'color13', 13: 'color14',
                           14: 'color15', 15: 'color16'}
            for e in g.edges():
                c = g[e[0]][e[1]]['color']
                if color_names[c] not in self.colored_edges.keys():
                    self.colored_edges[color_names[c]] = [list(e)]
                else:
                    self.colored_edges[color_names[c]].append(list(e))
        return self.colored_edges

    def get_nx_graph(self):
        nx_graph = {}
        for i in self.topology.keys():
            nx_graph[i] = [y for x in self.topology[i] for y in x if y != i]
        g = nx.from_dict_of_lists(nx_graph)
        return g

    def get_savepath(self):
        return os.path.join(self.filepath, self.name)


class TFI(Hamiltonian):
    def __init__(self, topology: Union[dict, str], g: float = 1.0, **kwargs):
        r"""
        The transverse field Ising-model is given by the Hamiltonian

        .. math::

            H = -\sum_{<i,j>}^N \sigma_{i}^{z}\sigma_{j}^{z} - g \sum_{i}^N \sigma_{i}^{x}

        with :math:`N` the number of spins. This function takes kwargs ``L`` and ``M`` that can be used to specify the
        size of the standard topologies ['line', 'rect_lattice'].
        
        Args:
            *topology (dict, str)*: 
                A dict with nodes as keys and a list of edges as values or a string defining a standard topology
            
            *g (float)*: 
                order parameter for the transverse field Ising-model
            
            *\*\*kwargs*:
                Additional arguments.
            
        Returns (inplace):
            None
      
        """
        assert isinstance(topology,
                          (dict, str)), "Topology must be a string or a dict, received object of type {}".format(
            type(topology))

        if isinstance(topology, str):
            assert 'L' in kwargs.keys(), "If topology is a string, the lattice or line size 'L' must be supplied as a kwarg"
            L = kwargs.pop('L')
            topology = standard_topologies(L, topology=topology, **kwargs)

        # TFI model #
        topology = remove_double_counting(topology)
        interactions = {'zz': topology, 'x': magnetic_field_interaction(topology)}
        f_or_af = kwargs.pop('f_or_af', 'f')
        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1
        name = kwargs.pop('name', "TFI_{}qb_g_{:.2f}".format(self.nsites, g))
        if f_or_af == 'f':
            model_parameters = {'zz': -1.0, 'x': -g}
        else:
            model_parameters = {'zz': 1.0, 'x': g}
        if 'boundary_conditions' in kwargs.keys():
            name = name + '_' + kwargs['boundary_conditions'] + '_' + f_or_af
        else:
            name = name + '_' + f_or_af
        additional_interactions = kwargs.pop("additional_interactions", {})
        additional_model_parameters = kwargs.pop("additional_model_parameters", {})
        interactions = {**interactions, **additional_interactions}
        model_parameters = {**model_parameters, **additional_model_parameters}

        super(TFI, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)


class XY(Hamiltonian):
    def __init__(self, topology: Union[dict, str], g: float = 1.0, gamma: float = 1.0, **kwargs):
        r"""
        The XY model with transverse field is given by the Hamiltonian

        .. math::

            H = -\sum_{<i,j>}^N \left( \frac{1+\gamma}{2} \sigma_{i}^{z}\sigma_{j}^{z} +
            \frac{1-\gamma}{2} \sigma_{i}^{z}\sigma_{j}^{z} \right) - g \sum_{i}^N \sigma_{i}^{z}

        with :math:`N` the number of spins. This function takes kwargs ``L`` and ``M`` that can be used to specify the
        size of the standard topologies ['line', 'rect_lattice'].

        Args:
            *topology (dict, str)*:
                A dict with nodes as keys and a list of edges as values or a string defining a standard topology

            *g (float)*:
                order parameter for the transverse field Ising-model

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """
        assert isinstance(topology,
                          (dict, str)), "Topology must be a string or a dict, received object of type {}".format(
            type(topology))

        if isinstance(topology, str):
            assert 'L' in kwargs.keys(), "If topology is a string, the lattice or line size 'L' must be supplied as a kwarg"
            L = kwargs.pop('L')
            topology = standard_topologies(L, topology=topology, **kwargs)

        # TFI model #
        topology = remove_double_counting(topology)
        interactions = {'xx': topology, 'yy': topology, 'z': magnetic_field_interaction(topology)}
        f_or_af = kwargs.pop('f_or_af', 'f')
        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1
        name = kwargs.pop('name', "XY_{}qb_gamma_{:.2f}_g_{:.2f}".format(self.nsites, gamma, g))
        if f_or_af == 'f':
            model_parameters = {'xx': -(1.0 - gamma) / 2, 'yy': -(1.0 + gamma) / 2, 'z': -g}
        else:
            model_parameters = {'xx': (1.0 - gamma) / 2, 'yy': (1.0 + gamma) / 2, 'z': g}
        if 'boundary_conditions' in kwargs.keys():
            name = name + '_' + kwargs['boundary_conditions'] + '_' + f_or_af
        else:
            name = name + '_' + f_or_af
        additional_interactions = kwargs.pop("additional_interactions", {})
        additional_model_parameters = kwargs.pop("additional_model_parameters", {})
        interactions = {**interactions, **additional_interactions}
        model_parameters = {**model_parameters, **additional_model_parameters}

        super(XY, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)


class HeisenbergXXX(Hamiltonian):
    def __init__(self, topology: Union[dict, str], **kwargs):
        r"""
        The XXX Heisenberg model is given by the Hamiltonian

        .. math::

            H = \sum_{<i,j>}^N \sigma_{i}^{x}\sigma_{j}^{x} + \sigma_{i}^{y}\sigma_{j}^{y}
            + \sigma_{i}^{z}\sigma_{j}^{z}

        with :math:`N` the number of spins. This function takes kwargs ``L`` and ``M`` that can be used to specify the
        size of the standard topologies ['line', 'rect_lattice'].

        Args:
            *topology (dict, str)*:
                A dict with nodes as keys and a list of edges as values or a string defining a standard topology

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """
        assert isinstance(topology,
                          (dict, str)), "Topology must be a string or a dict, received object of type {}".format(
            type(topology))

        if isinstance(topology, str):
            assert 'L' in kwargs.keys(), "If topology is a string, the lattice or line size 'L' must be supplied as a kwarg"
            L = kwargs.pop('L')
            topology = standard_topologies(L, topology=topology, **kwargs)
        topology = remove_double_counting(topology)
        f_or_af = kwargs.pop('f_or_af', 'f')
        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1
        name = kwargs.pop('name', "XXX_{}qb".format(self.nsites))
        if f_or_af == 'f':
            model_parameters = {'xx': -1.0, 'yy': -1.0, 'zz': -1.0}
        else:
            model_parameters = {'xx': 1.0, 'yy': 1.0, 'zz': 1.0}
        if 'boundary_conditions' in kwargs.keys():
            name = name + '_' + kwargs['boundary_conditions'] + '_' + f_or_af
        else:
            name = name + '_' + f_or_af
        # Heisenberg XXX model #
        interactions = {'xx': topology, 'yy': topology, 'zz': topology}

        additional_interactions = kwargs.pop("additional_interactions", {})
        additional_model_parameters = kwargs.pop("additional_model_parameters", {})
        interactions = {**interactions, **additional_interactions}
        model_parameters = {**model_parameters, **additional_model_parameters}

        super(HeisenbergXXX, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)


class HeisenbergXXZ(Hamiltonian):

    def __init__(self, topology: Union[dict, str], delta: float = 1.0, **kwargs):
        r"""
        The XXZ Heisenberg model is given by the Hamiltonian

        .. math::

            H = \sum_{<i,j>}^N \sigma_{i}^{x}\sigma_{j}^{x} + \sigma_{i}^{y}\sigma_{j}^{y}
            + \Delta \sigma_{i}^{z}\sigma_{j}^{z}

        with :math:`N` the number of spins. This function takes kwargs ``L`` and ``M`` that can be used to specify the
        size of the standard topologies ['line', 'rect_lattice'].

        Args:
            *topology (dict, str)*:
                A dict with nodes as keys and a list of edges as values or a string defining a standard topology

            *delta (float)*:
                The order parameter.

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """
        assert isinstance(topology,
                          (dict, str)), "Topology must be a string or a dict, received object of type {}".format(
            type(topology))

        if isinstance(topology, str):
            assert 'L' in kwargs.keys(), "If topology is a string, the lattice or line size 'L' must be supplied as a kwarg"
            L = kwargs.pop('L')
            topology = standard_topologies(L, topology=topology, **kwargs)
        topology = remove_double_counting(topology)

        f_or_af = kwargs.pop('f_or_af', 'f')
        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1
        name = kwargs.pop('name', "XXZ_{}qb_delta_{:1.2f}".format(self.nsites, delta))
        if f_or_af == 'f':
            model_parameters = {'xx': -1.0, 'yy': -1.0, 'zz': -1.0}
        else:
            model_parameters = {'xx': 1.0, 'yy': 1.0, 'zz': 1.0}
        if 'boundary_conditions' in kwargs.keys():
            name = name + '_' + kwargs['boundary_conditions'] + '_' + f_or_af
        else:
            name = name + '_' + f_or_af

        # Heisenberg XXZ model
        interactions = {'xx': topology, 'yy': topology, 'zz': topology}
        model_parameters['zz'] = delta

        additional_interactions = kwargs.pop("additional_interactions", {})
        additional_model_parameters = kwargs.pop("additional_model_parameters", {})
        interactions = {**interactions, **additional_interactions}
        model_parameters = {**model_parameters, **additional_model_parameters}
        super(HeisenbergXXZ, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)


class HaldaneShastry(Hamiltonian):

    def __init__(self, L, modified=False, **kwargs):
        r"""
        The Haldane-Shastry model on a chain is given by the Hamiltonian

        .. math::

            H = \sum_{j<k}^N\frac{1}{d^2_{jk}} (+\sigma_{i}^{x}\sigma_{j}^{x} + \sigma_{i}^{y}\sigma_{j}^{y}
            + \sigma_{i}^{z}\sigma_{j}^{z})

        with :math:`d_{jk}=\frac{N}{\pi}|\sin[\pi(i-j)/N]|` and :math:`N` the number
        of spins. This function takes kwargs ``L`` and ``M`` that can be used to specify the
        size of the standard topologies ['line', 'rect_lattice'].
            *L*:
                Length of the chain. Must be even.
            *modified*:
                Boolean indicating if the X and Y interactions have a negative sign.
            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None[x

        """

        def chord_distance(link, sign):
            d = L / np.pi * np.abs(np.sin(np.pi * (link[1] - link[0]) / L))
            return sign * 1 / d ** 2

        assert L / 2 == L // 2, "N must be even, received {}".format(L)
        topology = fully_connected(L)
        topology = remove_double_counting(topology)
        interactions = {'xx': topology, 'yy': topology, 'zz': topology}
        links = [tuple(y) for x in topology.values() for y in x]
        if modified:
            sign = -1
        else:
            sign = 1
        random_xx = dict(zip(links, map(chord_distance, links, list(sign for _ in range(len(links))))))
        random_yy = dict(zip(links, map(chord_distance, links, list(sign for _ in range(len(links))))))
        random_zz = dict(zip(links, map(chord_distance, links, list(+1 for _ in range(len(links))))))

        model_parameters = {'xx': random_xx, 'yy': random_yy, 'zz': random_zz}

        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1
        if modified:
            name = kwargs.pop('name', "MHS_{}qb".format(self.nsites))
        else:
            name = kwargs.pop('name', "HS_{}qb".format(self.nsites))

        additional_interactions = kwargs.pop("additional_interactions", {})
        additional_model_parameters = kwargs.pop("additional_model_parameters", {})
        interactions = {**interactions, **additional_interactions}
        model_parameters = {**model_parameters, **additional_model_parameters}
        super(HaldaneShastry, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)


class KitaevHoneycomb(Hamiltonian):
    def __init__(self, L, Jxx, Jyy, Jzz, f_or_af, **kwargs):
        r"""
        The Kitaev Honeycomb model is given by the Hamiltonian

        .. math::

            H = -\sum_{<i,j>}^N J_{xx}\sigma_{i}^{x}\sigma_{j}^{x} - J_{yy}\sigma_{i}^{y}\sigma_{j}^{y}
            - J_{zz}\sigma_{i}^{z}\sigma_{j}^{z}

        with :math:`N` the number of spins.

        Args:
            *L (int)*:
                L defines the number of spins in the honeycomb. Supported topologies exist for L=10,13

            *Jxx (float)*:
                The order parameter controlling the strength of the :math:`X-X` interactions.

            *Jyy (float)*:
                The order parameter controlling the strength of the :math:`Y-Y` interactions.

            *Jzz (float)*:
                The order parameter controlling the strength of the :math:`Z-Z` interactions.

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """
        supported_system_sizes = [8, 16]
        assert L in supported_system_sizes, "Supported systems sizes are {}, received L={}".format(
            supported_system_sizes, L)
        if L == 8:
            topology = graph_honeycomb_8()
            interactions = {'xx': graph_honeycomb_8('xx'),
                            'yy': graph_honeycomb_8('yy'),
                            'zz': graph_honeycomb_8('zz')}

        elif L == 16:
            topology = graph_honeycomb_16()
            interactions = {'xx': graph_honeycomb_16('xx'),
                            'yy': graph_honeycomb_16('yy'),
                            'zz': graph_honeycomb_16('zz')}
        topology = remove_double_counting(topology)
        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1
        name = kwargs.pop('name',
                          "Kitaev_honeycomb_{}qb_Jaa_{:1.2f}_{:1.2f}_{:1.2f}".format(self.nsites, Jxx, Jyy, Jzz))

        if f_or_af == 'f':
            model_parameters = {'xx': -Jxx, 'yy': -Jyy, 'zz': -Jzz}
        else:
            model_parameters = {'xx': Jxx, 'yy': Jyy, 'zz': Jzz}
        if 'boundary_conditions' in kwargs.keys():
            name = name + '_' + kwargs['boundary_conditions'] + '_' + f_or_af
        else:
            name = name + '_' + f_or_af
        # Heisenberg XYZ model #

        additional_interactions = kwargs.pop("additional_interactions", {})
        additional_model_parameters = kwargs.pop("additional_model_parameters", {})
        interactions = {**interactions, **additional_interactions}
        model_parameters = {**model_parameters, **additional_model_parameters}

        super(KitaevHoneycomb, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)


class KitaevLadder(Hamiltonian):
    def __init__(self, L, Jxx, Jyy, Jzz, f_or_af, **kwargs):
        r"""
        The Kitaev Honeycomb model is given by the Hamiltonian

        .. math::

            H = -\sum_{<i,j>}^N J_{xx}\sigma_{i}^{x}\sigma_{j}^{x} - J_{yy}\sigma_{i}^{y}\sigma_{j}^{y}
            - J_{zz}\sigma_{i}^{z}\sigma_{j}^{z}

        with :math:`N` the number of spins.

        Args:
            *L (int)*:
                L defines the number of spins in the honeycomb. Supported topologies exist for L=10,13

            *Jxx (float)*:
                The order parameter controlling the strength of the :math:`X-X` interactions.

            *Jyy (float)*:
                The order parameter controlling the strength of the :math:`Y-Y` interactions.

            *Jzz (float)*:
                The order parameter controlling the strength of the :math:`Z-Z` interactions.

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """

        topology = graph_ladder(L, boundary_condition=kwargs['boundary_conditions'])
        interactions = {'xx': graph_ladder(L, 'xx', kwargs['boundary_conditions']),
                        'yy': graph_ladder(L, 'yy', kwargs['boundary_conditions']),
                        'zz': graph_ladder(L, 'zz', kwargs['boundary_conditions'])}

        topology = remove_double_counting(topology)
        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1
        name = kwargs.pop('name',
                          "Kitaev_honeycomb_{}qb_Jaa_{:1.2f}_{:1.2f}".format(self.nsites, Jxx, Jzz))

        if f_or_af == 'f':
            model_parameters = {'xx': -Jxx, 'yy': -Jyy, 'zz': -Jzz}
        else:
            model_parameters = {'xx': Jxx, 'yy': Jyy, 'zz': Jzz}
        if 'boundary_conditions' in kwargs.keys():
            name = name + '_' + kwargs['boundary_conditions'] + '_' + f_or_af
        else:
            name = name + '_' + f_or_af

        additional_interactions = kwargs.pop("additional_interactions", {})
        additional_model_parameters = kwargs.pop("additional_model_parameters", {})
        interactions = {**interactions, **additional_interactions}
        model_parameters = {**model_parameters, **additional_model_parameters}

        super(KitaevLadder, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)


class HeisenbergXYZ(Hamiltonian):
    def __init__(self, topology: Union[dict, str], delta, J, **kwargs):
        r"""
        The XYZ Heisenberg model is given by the Hamiltonian

        .. math::

            H = \sum_{<i,j>}^N \sigma_{i}^{x}\sigma_{j}^{x} + J \sigma_{i}^{y}\sigma_{j}^{y}
            + \Delta \sigma_{i}^{z}\sigma_{j}^{z}

        with :math:`N` the number of spins. This function takes kwargs ``L`` and ``M`` that can be used to specify the
        size of the standard topologies ['line', 'rect_lattice'].

        Args:
            *topology (dict, str)*:
                A dict with nodes as keys and a list of edges as values or a string defining a standard topology

            *delta (float)*:
                The order parameter controlling the strength of the :math:`Z-Z` interactions.

            *J (float)*:
                The order parameter controlling the strength of the :math:`Y-Y` interactions.

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """
        assert isinstance(topology,
                          (dict, str)), "Topology must be a string or a dict, received object of type {}".format(
            type(topology))

        if isinstance(topology, str):
            assert 'L' in kwargs.keys(), "If topology is a string, the lattice or line size 'L' must be supplied as a kwarg"
            L = kwargs.pop('L')
            topology = standard_topologies(L, topology=topology, **kwargs)
        topology = remove_double_counting(topology)

        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1
        name = kwargs.pop('name',
                          "XYZ_{}qb_delta_{:1.2f}_J_{:1.2f}".format(self.nsites, delta, J))

        if 'boundary_conditions' in kwargs.keys():
            name = name + '_' + kwargs['boundary_conditions']
        self.boundary_conditions = kwargs.pop('boundary_conditions', None)

        print(name)
        # Heisenberg XYZ model #
        interactions = {'xx': topology, 'yy': topology, 'zz': topology}
        model_parameters = {'yy': delta, 'zz': J}

        additional_interactions = kwargs.pop("additional_interactions", {})
        additional_model_parameters = kwargs.pop("additional_model_parameters", {})
        interactions = {**interactions, **additional_interactions}
        model_parameters = {**model_parameters, **additional_model_parameters}

        super(HeisenbergXYZ, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)


class RandomFullyConnectedXYZ(Hamiltonian):
    def __init__(self, L, seed: int = 1337, **kwargs):
        r"""
        The fully-connected random couplings is given by the Hamiltonian

        .. math::

            H = \sum_{ij\alpha} w_{ij}^{\alpha} \sigma_{i}^{\alpha}\sigma_{j}^{\alpha}

        with :math:`\alpha,x,y,z` and :math:`N` the number of spins. By default, the interactions are sampled
        from a gaussian :math:`w_{ij}^{\alpha} \sim \mathcal{N}(0,1)`. However, a custom random number generator can
        be supplied through the kwarg ``rng``.

        Args:
            *L (int)*:
                An integer defining the number of vertices.

            *seed (int)*:
                Seed for the random number generator defining the couplings

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """

        topology = fully_connected(L)
        topology = remove_double_counting(topology)
        rng = kwargs.pop('rng', np.random.RandomState(seed).randn)
        assert callable(rng), "Random number generator rng must be a callable, received {}".format(type(rng))
        # Heisenberg XYZ model #
        interactions = {'xx': topology, 'yy': topology, 'zz': topology}
        links = [tuple(y) for x in topology.values() for y in x]
        random_xx = dict(zip(links, rng(len(links))))
        random_yy = dict(zip(links, rng(len(links))))
        random_zz = dict(zip(links, rng(len(links))))
        model_parameters = {'xx': random_xx, 'yy': random_yy, 'zz': random_zz}

        additional_interactions = kwargs.pop("additional_interactions", {})
        additional_model_parameters = kwargs.pop("additional_model_parameters", {})

        interactions = {**interactions, **additional_interactions}
        model_parameters = {**model_parameters, **additional_model_parameters}
        assert len(interactions) == len(
            model_parameters), "The number of interactions and model_parameters is not the same, received".format(
            interactions.keys(), model_parameters.keys())
        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1
        name = kwargs.pop('name', "Random_XYZ_{}qb_seed_{}".format(self.nsites, seed))

        super(RandomFullyConnectedXYZ, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)


class QuantumBoltzmann(Hamiltonian):
    def __init__(self, L, seed: int = 1337, **kwargs):
        r"""
        The fully-connected random couplings and fields model is given by the Hamiltonian

        .. math::

            H = \sum_{ij\alpha} w_{ij}^{\alpha} \sigma_{i}^{\alpha}\sigma_{j}^{\alpha} +  \sum_{i\alpha} h_{i}^{\alpha} \sigma_{i}^{\alpha}

        with :math:`\alpha,x,y,z` and :math:`N` the number of spins. By default, the interactions are sampled
        from a gaussian :math:`h_i^\alpha,w_{ij}^{\alpha} \sim \mathcal{N}(0,1)`. However, a custom random number generator can
        be supplied through the kwarg ``rng``.

        Args:
            *L (int)*:
                An integer defining the number of vertices.

            *seed (int)*:
                Seed for the random number generator defining the couplings

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """

        topology = fully_connected(L)
        topology = remove_double_counting(topology)
        all_edges = {tuple(sorted(x)) for y in topology.values() for x in y}
        self.nsites = max(all_edges, key=itemgetter(1))[1] + 1
        np.random.seed(seed)

        # Heisenberg XYZ model #
        mag_field = magnetic_field_interaction(topology)
        interactions = {'xx': topology, 'yy': topology, 'zz': topology,
                        'x': mag_field, 'y': mag_field, 'z': mag_field}
        links = [tuple(y) for x in topology.values() for y in x]
        sites = [(s,) for s in range(self.nsites)]
        random_xx = dict(zip(links, np.random.randn(len(links))))
        random_yy = dict(zip(links, np.random.randn(len(links))))
        random_zz = dict(zip(links, np.random.randn(len(links))))
        random_x = dict(zip(sites, np.random.randn(len(sites))))
        random_y = dict(zip(sites, np.random.randn(len(sites))))
        random_z = dict(zip(sites, np.random.randn(len(sites))))
        model_parameters = {'xx': random_xx, 'yy': random_yy, 'zz': random_zz,
                            'x': random_x, 'y': random_y, 'z': random_z}
        additional_interactions = kwargs.pop("additional_interactions", {})
        additional_model_parameters = kwargs.pop("additional_model_parameters", {})

        interactions = {**interactions, **additional_interactions}
        model_parameters = {**model_parameters, **additional_model_parameters}
        assert len(interactions) == len(
            model_parameters), "The number of interactions and model_parameters is not the same, received".format(
            interactions.keys(), model_parameters.keys())

        name = kwargs.pop('name', "QBM_{}_spins_seed_{}".format(self.nsites, seed))

        super(QuantumBoltzmann, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)


class J1J2(Hamiltonian):
    def __init__(self, topology, J1, J2, **kwargs):
        r"""
        The :math:`J_1-J_2` model is given by the Hamiltonian

        .. math::

            H = J_1 \sum_{<i,j>}^N \vec{\sigma}_i \cdot \vec{\sigma}_j + J_2 \sum_{<<i,j>>}^N \vec{\sigma}_i \cdot \vec{\sigma}_j

        with :math:`N` the number of spins. This function takes kwargs ``L`` and ``M`` that can be used to specify the
        size of the standard topologies ['line', 'rect_lattice'].

        Args:
            *L (int)*:
                An integer defining the number of vertices.

            *J1 (float)*:
                The order parameter controlling the strength of the nearest neighbour interactions.

            *J2 (float)*:
                The order parameter controlling the strength of the nearest-nearest neighbour interactions.

            *\*\*kwargs*:
                Additional arguments.

        Returns (inplace):
            None

        """
        assert isinstance(topology,
                          (dict, str)), "Topology must be a string or a dict, received object of type {}".format(
            type(topology))
        self.boundary_conditions = kwargs.pop('boundary_conditions', None)

        # if isinstance(topology, str):
        #     assert 'L' in kwargs.keys(), "If topology is a string, the lattice or line size 'L' must be supplied as a kwarg"
        #     L = kwargs.pop('L')
        #     topology = standard_topologies(L, topology=topology, **kwargs)
        # topology = remove_double_counting(topology)

        # # Heisenberg XYZ model #
        # interactions = {'xx': topology, 'yy': topology, 'zz': topology}
        # model_parameters = {'xx': J1, 'yy': delta, 'zz': J}
        #
        # additional_interactions = kwargs.pop("additional_interactions", {})
        # additional_model_parameters = kwargs.pop("additional_model_parameters", {})
        # interactions = {**interactions, **additional_interactions}
        # model_parameters = {**model_parameters, **additional_model_parameters}
        # all_edges = {tuple(sorted(x)) for y in topology.values() for x in y }
        # self.nsites = max(all_edges,key=itemgetter(1))[1] + 1
        # name = kwargs.pop('name', "XYZ_{}qb_delta_{}_J".format(self.nsites, delta, J))
        # super(J1J2, self).__init__(topology, interactions, model_parameters, name=name, **kwargs)

        raise NotImplementedError


def remove_double_counting(g: dict) -> dict:
    r"""
    Removes the double counted edges in a graph :math:`\mathcal{G}:=(N, G)`.

    Args:
        *g (dict)*:
            A dict with nodes as keys and a list of edges as values.

    Returns (dict):
        A dict with nodes as keys and a list of edges as values.
    """
    single_edge_g = {}
    if all(isinstance(l, int) for l in g.values()):
        all_edges = {tuple(sorted(x)) for x in g.values()}
    else:
        all_edges = {tuple(sorted(y)) for x in g.values() for y in x}
    new_keys = {x[0] for x in all_edges}
    # new_keys.add(max(all_edges, key=itemgetter(1))[1])
    for k in new_keys:
        single_edge_g[k] = [list(x) for x in all_edges if x[0] == k]
    return single_edge_g


def fully_connected(L: int) -> dict:
    r"""
    Get a fully connected graph :math:`\mathcal{G}:=(N, G)` of L vertices.

    Args:
        *L (int)*:
            The number of vertices in the graph.

    Returns (dict):
        A dict with nodes as keys and a list of edges as values.
    """
    top = {}
    for site in range(L):
        top[site] = [[site, other_site] for other_site in range(L) if other_site != site]
    return top


def standard_topologies(L, topology: str, **kwargs) -> dict:
    """
    Get a dictionary of a standard 1D or 2D topology, such as a line or square lattice.
    Pass boundary_conditions as a kwarg to specifiy the boundary condition of the standard topology.

    Args:
        *topology (str)*:
            String defining the topology. For now, only 'line' and 'rect_lattice' are supported.

        *\*\*kwargs*:
            Additional arguments.

    Returns (dict):
        A dict with nodes as keys and a list of edges as values.
    """
    assert topology in ['line', 'rect_lattice'], "topology must be 'line', 'rect_lattice', received {}".format(
        topology)
    top = {}
    M = kwargs.pop("M", L)
    boundary_conditions = kwargs.pop('boundary_conditions', 'open')
    if topology == 'line':
        assert boundary_conditions in ['closed', 'open'], \
            'boundary conditions must be "closed" or "open" for "line", received {}'.format(
                boundary_conditions)
        top[0] = [[0, 1]]
        for s in range(1, L - 1):
            top[s] = [[s, s - 1], [s, s + 1]]
        top[L - 1] = [[L - 1, L - 2]]
        if boundary_conditions == 'closed':
            top[L - 1] = [[L - 1, 0]]
    elif topology == 'rect_lattice':
        assert boundary_conditions in ['open', 'torus', 'cylinder'], \
            'boundary conditions must be "open", "torus", "cylinder" for 2D, received {}'.format(
                boundary_conditions)
        nx_graph = {}
        if boundary_conditions == 'open':
            nx_graph = nx.to_dict_of_lists(nx.grid_2d_graph(L, M))
        elif boundary_conditions == 'torus':
            nx_graph = nx.to_dict_of_lists(nx.grid_2d_graph(L, M, periodic=True))
        elif boundary_conditions == 'cylinder':
            nx_graph = nx.to_dict_of_lists(nx.grid_2d_graph(L, M, periodic=True))
            for k, v in nx_graph.items():
                if k[0] == 0:
                    nx_graph[k].remove((L - 1, k[1]))

                if k[0] == M - 1:
                    nx_graph[k].remove((0, k[1]))
        for k, v in nx_graph.items():
            unraveled_site = int(np.ravel_multi_index(k, dims=(L, M)))
            top[unraveled_site] = [[unraveled_site, np.ravel_multi_index(edge, dims=(L, M))] for edge in v]
    return top


def magnetic_field_interaction(topology):
    r"""
    Add a magnetic field interaction in the :math:`\alpha` direction at each site in the topology.

    .. math::

        H_{\text{mag}} = \sum_{i} h_i^\alpha \sigma_i^\alpha

    Args:
        *topology (dict)*:
            A dict with nodes as keys and a list of edges as values.

    Returns (dict):
        A dict with nodes as keys and a list of edges as values.

    """
    # Since we order the interactions from small vertex to large vertex, the largerst vertex will not show up in the
    # keys() dict, which is why we add it here.
    sites = list(topology.keys()) + [max(topology.keys()) + 1]
    return dict(zip(sites, [[[k]] for k in sites]))
