import numpy as np

def quad_mu12(k1, k2, n):
    x, wx = np.polynomial.legendre.leggauss(n)
    # the 1D mapping
    mu12_1d = 0.5 * (x+1)**2 - 1
    w_1d = (x+1) * wx
    k3_1d = np.sqrt(k1**2 + k2**2 + 2*k1*k2*mu12_1d)
    
    # exact k3 mapping
    k3_exact = min(k1, k2) * x + max(k1, k2)
    w_exact = min(k1, k2) / (k1 * k2) * k3_exact * wx
    # wait, dmu = w_exact dx
    return np.sum(w_1d / k3_1d**3), np.sum(w_exact / k3_exact**3)

k1, k2 = 0.1, 0.12
print("k1=0.1, k2=0.12")
for n in [5, 10, 20]:
    v1, v2 = quad_mu12(k1, k2, n)
    print(f"n={n}: 1D={v1:.4f}, exact k3={v2:.4f}")

k1, k2 = 0.1, 0.1001
print("\nk1=0.1, k2=0.1001")
for n in [5, 10, 20, 60, 100]:
    v1, v2 = quad_mu12(k1, k2, n)
    print(f"n={n}: 1D={v1:.4f}, exact k3={v2:.4f}")
