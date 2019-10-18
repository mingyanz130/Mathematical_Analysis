# linear_systems.py
"""Volume 1: Linear Systems.
<Mingyan Zhao>
<Math 345>
<10/08/2018>
"""
import numpy as np
import time
from scipy import linalg as la
from random import random
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as spla
# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    #find the dimension of the input
    A = A.astype(float)
    n = A.shape[0]
    #repeat theGaussian Elmination on lower triangal
    for j in range(n-1):
        for i in range(j + 1,n):
            A[i,j:] -= (A[i,j]/A[j,j])*A[j,j:]
    return A

# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    #change the data type
    A = A.astype(float)
    #find the dimension
    n = A.shape[0]
    #initialize L, U
    U = np.copy(A)
    L = np.eye(n)
    #repeat the algorithm given
    for j in range(n):
        for i in range(j+1, n):
            L[i,j] = U[i,j]/U[j,j]
            U[i,j:] -=  L[i,j]*U[j,j:]
    return L, U

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    #change the data type
    A = A.astype(float)
    #find the dimension
    n = A.shape[0]
    #find L, U
    L,U = lu(A)
    #initialize X, Y
    Y = np.zeros(n)
    X = np.zeros(n)
    #claculation for Y
    for i in range(n):
        summation = 0
        
        for j in range(i):
            summation += L[i][j]*Y[j]
            
        Y[i] = b[i] - summation
     
    
    #claculation for X    
    for i in reversed(range(n)):
        summation = 0
        
        for j in range(i, n):
            summation += U[i][j]*X[j]/U[i][i]
            
        X[i] = (Y[i]/U[i][i] - summation)
        
    print(U@X - Y)
        
    return X
# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """    
    #set the domains
    domain = 2**np.arange(1,10)
    
    times1 = []
    times2 = []
    times3 = []
    times4 = []
    
    #record the start time and end time for each function
    for n in domain:
        A = np.random.random((n,n))
        b = np.random.random(n)
        
        start1 = time.time()
        A_ = la.inv(A)
        X = A_@b
        times1.append(time.time() -start1)
        
        start2 = time.time()
        la.solve(A, b)
        times2.append(time.time() -start2)
        
        start3 = time.time()
        L, P = la.lu_factor(A)
        x= la.lu_solve((L,P), b)
        times3.append(time.time() -start3)
        
        
        L, P = la.lu_factor(A)
        start4 = time.time()
        x= la.lu_solve((L,P), b)
        times4.append(time.time() -start4)
        
        
    # plot all the   
    
    
    plt.loglog(domain, times1, 'g.-', basex = 2, basey = 2, linewidth = 2, markersize = 15, label = "Invert A")
    plt.loglog(domain, times2, 'b.-', basex = 2, basey = 2, linewidth = 2, markersize = 15, label = "la.solve")
    plt.loglog(domain, times3, 'c.-', basex = 2, basey = 2, linewidth = 2, markersize = 15, label = "la.lu_factor()")
    plt.loglog(domain, times4, 'y.-', basex = 2, basey = 2, linewidth = 2, markersize = 15, label = "la.lu_solve only")
    plt.legend(loc = "upper left")
    plt.show()

# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    #initialize B
    B = sparse.diags([1,-4,1],[-1,0,1], shape=(n,n))
    
    #set up A with B and I
    A = sparse.block_diag([B]*n)
    A.setdiag(1,n)
    A.setdiag(1,-n)
    
   
    return A

# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    time1 = []
    time2 = []
    domain = 2**np.arange(1,7)
    for n in domain:    
        #initialize A and b
        A = prob5(n)
        b = np.random.random(n**2)
      
        #record time for each n for both methods
        start1 = time.time()
        Acsr = A.tocsr()
        spla.spsolve(Acsr,b)
        time1.append(time.time() -start1)
        
        start2 = time.time()
        A_np = A.toarray()
        la.solve(A_np, b)
        time2.append(time.time() -start2)
    
    #plot the graph 
    plt.loglog(domain, time1, 'g.-', basex = 2, basey = 2, linewidth = 2, markersize = 15, label = "Convert A to CSR") 
    plt.loglog(domain, time2, 'b.-', basex = 2, basey = 2, linewidth = 2, markersize = 15, label = "Convert A to numpy") 
    plt.legend()
    plt.show()