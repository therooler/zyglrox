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

from random import choice, randint, seed
seed(1337)

# Code directly from A Simple and Fast Heuristic Algorithm for Edge-coloring of Graphs, M.A. Fiol, J. Vilaltella 2013

def properlyColored(G, u, D):
    return len(set([color['color'] for color in G[u].values()])) == G.degree(u) and all(color['color'] in range(D) for color in G[u].values())


def checkEdgeColoring(G, D):
    return all(properlyColored(G, u, D) for u in G.nodes())


def conflictLevel(G, u):
    return G.degree(u) - len(set([color['color'] for color in G[u].values()]))


def createConflictDictionary(G, D):
    conflict_dictionary = dict([(i, set([])) for i in range(1, D)])
    for u in G.nodes():
        conflict_level_u = conflictLevel(G, u)
        if conflict_level_u > 0: conflict_dictionary[conflict_level_u].add(u)
    return conflict_dictionary


def updateConflictDictionary(G, u, conflict_dictionary, old_conflict_level_u):
    conflict_level_u = conflictLevel(G, u)
    if old_conflict_level_u > 0:
        conflict_dictionary[old_conflict_level_u].remove(u)
    if conflict_level_u > 0:
        conflict_dictionary[conflict_level_u].add(u)
    return conflict_level_u - old_conflict_level_u


def maxConflictLevel(conflict_dictionary):
    return max(
        [conflict_level for conflict_level in conflict_dictionary if len(conflict_dictionary[conflict_level]) > 0])


def totalNumberOfConflicts(conflict_dictionary):
    return sum(conflict_level * len(conflict_dictionary[conflict_level]) for conflict_level in conflict_dictionary)


def colorEdgeAndUpdate(G, u, v, color, conflict_dictionary):
    old_conflict_level_u = conflictLevel(G, u)
    old_conflict_level_v = conflictLevel(G, v)
    G[u][v]['color'] = G[v][u]['color'] = color
    updateConflictDictionary(G, u, conflict_dictionary, old_conflict_level_u)
    return updateConflictDictionary(G, v, conflict_dictionary, old_conflict_level_v)


def KempeNext(G, last, node, new_color, conflict_dictionary):
    available_for_next = [w for w in G[node] if w != last and G[node][w]['color'] == new_color]
    if available_for_next == []:
        next_node = None
    else:
        next_node = choice(available_for_next)
    old_color = G[last][node]['color']
    conflict_level_variation = colorEdgeAndUpdate(G, last, node, new_color, conflict_dictionary)
    return conflict_level_variation, old_color, next_node


def KempeStep(G, last, node, new_color, conflict_dictionary):
    conflict_level_variation, old_color, next_node = KempeNext(G, last, node, new_color, conflict_dictionary)
    if conflict_level_variation < 0 or next_node == None:
        return node, None, None
    return node, next_node, old_color


def KempeProcess(G, last, node, new_color, conflict_dictionary):
    Kempe_chain = set([])
    while new_color != None and last not in Kempe_chain:
        Kempe_chain.add(last)
        last, node, new_color = KempeStep(G, last, node, new_color, conflict_dictionary)


def KempeStart(G, D, node, conflict_dictionary):
    colors = set(range(D))
    next_node = None
    for adjacent in G[node]:
        edge_color = G[node][adjacent]['color']
        if edge_color in colors:
            colors.remove(edge_color)
        else:
            next_node = adjacent
    if next_node != None:
        KempeProcess(G, node, next_node, choice(list(colors)), conflict_dictionary)


def preColoring(G, D):  # Pre-coloring with a greedy algorithm
    for e in G.edges():
        G[e[0]][e[1]]['color'] = G[e[1]][e[0]]['color'] = None
    for e in G.edges():
        available_colors = set(range(D))
        available_colors -= set([y for x in G[e[0]].values() for y in x.values()])
        available_colors -= set([y for x in G[e[1]].values() for y in x.values()])
        if available_colors == set():
            G[e[0]][e[1]]['color'] = G[e[1]][e[0]]['color'] = randint(0, D - 1)
        else:
            G[e[0]][e[1]]['color'] = G[e[1]][e[0]]['color'] = choice(list(available_colors))


def heuristic(G, D, repetition_limit):
    repetitionCounter = 0
    conflict_dictionary = createConflictDictionary(G, D)
    previous = current = totalNumberOfConflicts(conflict_dictionary)
    while previous > 0:
        highest_conflict_level = maxConflictLevel(conflict_dictionary)
        node = choice(list(conflict_dictionary[highest_conflict_level]))
        KempeStart(G, D, node, conflict_dictionary)
        current = totalNumberOfConflicts(conflict_dictionary)
        if current == 0:
            return True
        if current >= previous:
            repetitionCounter += 1
            if repetitionCounter > repetition_limit:
                return False
        else:
            repetitionCounter = 0
        previous = min(previous, current)
    return True


def applyHeuristic(G, D, repetition_limit, iteration_limit):
    preColoring(G, D)
    number_of_iterations = 1
    while not heuristic(G, D, repetition_limit):
        if number_of_iterations > iteration_limit:
            break
        preColoring(G, D)
        number_of_iterations += 1
    print("Number of iterations: {}".format(number_of_iterations))
    print("Edge-coloring successful {}".format(checkEdgeColoring(G, D)))
    return checkEdgeColoring(G,D)


# if __name__ == "__main__":
#     G = nx.random_regular_graph(4, 24)
#     print(G.edges())
#     applyHeuristic(G, 4, 10, 10)
    # for e in G.edges():
    #     print(G[e[0]][e[1]]['color'])
