import matplotlib.pyplot as plt
import numpy as np

def Marchenko_Pastur(N, nu):
    X = nu* np.random.randn(int(2 ** (N / 2)), int(2 ** (N / 2)))
    Y = X @ np.transpose(X)
    Y /= np.trace(Y)
    return np.log(np.sort(np.linalg.eigvals(Y))[::-1])

for i in [0.1, 0.5, 1.0, 2, 5, 10]:
    plt.plot(Marchenko_Pastur(16, i))
plt.show()