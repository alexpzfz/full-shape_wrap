import time
import numpy as np

# Mocking the functions to measure pre-computation vs dynamic
def mock_dyn(n):
    mu12 = np.random.rand(n, 1, 5, 1)
    # mock spherical harmonic
    for i in range(10):
        y1 = mu12**2
    return y1
    
start = time.time()
r = mock_dyn(100000)
print("Dynamic (100k triangles):", time.time() - start)
from scipy.special import lpmv

def mock_dyn_lpmv(n):
    mu12 = np.random.rand(n, 1, 5, 1)
    
    t0 = time.time()
    for l in range(3):
        for m in range(l+1):
            y1 = lpmv(m, l, mu12)
    return time.time() - t0

print("Dynamic lpmv (100k triangles for l<=2):", mock_dyn_lpmv(100000))
from sympy.physics.wigner import wigner_3j

def mock_dyn_full(n):
    mu12 = np.random.rand(n, 1, 5, 1)
    y2 = np.random.rand(1, 5, 1, 10)
    
    t0 = time.time()
    res = np.zeros((n, 5, 5, 10))
    for L in range(3):
        for M in range(-L, L+1):
            y1 = lpmv(abs(M), L, mu12)
            res += y1 * y2
    return time.time() - t0
print("Full loop sum (100k):", mock_dyn_full(100000))

