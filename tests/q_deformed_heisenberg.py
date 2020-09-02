from zyglrox.core.hamiltonians import HeisenbergXYZ

ham = HeisenbergXYZ({0:[[0,1]], 1: [[1,0]]}, delta=0.1, J=1)
ham.get_hamiltonian()

print(ham.groundstates)