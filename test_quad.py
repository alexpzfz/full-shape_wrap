import numpy as np

k1 = np.array([0.1])
k2 = np.array([0.1])

# standard GL
n = 5
mu_std, w_std = np.polynomial.legendre.leggauss(n)
k3_std = np.sqrt(k1**2 + k2**2 + 2*k1*k2*mu_std)
# integrand ~ 1/k3 ... actually we want to see convergence of int 1/k3 dmu
# wait, actually let's just integrate 1/k3
exact = np.log((k1+k2)/np.abs(k1-k2)) / (k1*k2) if k1!=k2 else 2.0 / (k1**2) # actually limit is 2/k^2
print("Exact 1/k3 for k1=0.1, k2=0.1001:", np.log(0.2001/0.0001)/(0.1*0.1001))

def test_k1k2(k1, k2, n):
    mu_std, w_std = np.polynomial.legendre.leggauss(n)
    k3_std = np.sqrt(k1**2 + k2**2 + 2*k1*k2*mu_std)
    val_std = np.sum(w_std / k3_std)
    
    x, w_x = np.polynomial.legendre.leggauss(n)
    mu_new = 0.5 * (x + 1)**2 - 1.0
    w_new = w_x * (x + 1)
    k3_new = np.sqrt(k1**2 + k2**2 + 2*k1*k2*mu_new)
    val_new = np.sum(w_new / k3_new)
    
    return val_std, val_new

print("n=5", test_k1k2(0.1, 0.1001, 5))
print("n=40", test_k1k2(0.1, 0.1001, 40))

# another test function: 1/k3^3
def test_k1k2_3(k1, k2, n):
    mu_std, w_std = np.polynomial.legendre.leggauss(n)
    k3_std = np.sqrt(k1**2 + k2**2 + 2*k1*k2*mu_std)
    val_std = np.sum(w_std / k3_std**3)
    
    x, w_x = np.polynomial.legendre.leggauss(n)
    mu_new = 0.5 * (x + 1)**2 - 1.0
    w_new = w_x * (x + 1)
    k3_new = np.sqrt(k1**2 + k2**2 + 2*k1*k2*mu_new)
    val_new = np.sum(w_new / k3_new**3)
    return val_std, val_new

print("1/k3^3, n=5", test_k1k2_3(0.1, 0.1001, 5))
print("1/k3^3, n=40", test_k1k2_3(0.1, 0.1001, 40))

