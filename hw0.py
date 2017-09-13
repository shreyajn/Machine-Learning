import scipy.io
import numpy 
import matplotlib.pyplot as plt
from scipy import linalg as LA
mat = scipy.io.loadmat('/Users/shreyajain/Downloads/hw0data.mat')

m = mat["M"]
print m
m = numpy.matrix(m)
print numpy.shape(m)
s = m[[3],:]
print s
print m[:,[4]]
avg = m[:,4].mean()
print avg
hi = plt.hist(s)
plt.show()
prod = numpy.dot(m.T,m)
print prod
e_vals, e_vecs = LA.eig(prod)
print e_vals
e_vals = e_vals.argsort()
print e_vals