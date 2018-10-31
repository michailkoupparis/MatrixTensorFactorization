from MatrixFactorization import myNMF
import numpy as np

V = np.random.normal(0,1.44,40).reshape(4,10)
V = np.abs(V)
myNM= myNMF()
myNM.fit(V)
