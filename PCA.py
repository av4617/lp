from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig

# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print 'Matrix A: \n',A

# calculate the mean of each column
M = mean(A.T, axis=1)
print 'Mean of each column(M): ',M

# center columns by subtracting column means
C = A - M
print 'Center Columns (C = A - M): \n',C

# calculate covariance matrix of centered matrix
V = cov(C.T)
print 'Covariance Matrix: ',V

# eigen decomposition of covariance matrix
values, vectors = eig(V)
print 'Eigen Vectors: ',vectors
print 'Eigen Values: ',values

# project data
P = vectors.T.dot(C.T)
print 'project data',P.T
