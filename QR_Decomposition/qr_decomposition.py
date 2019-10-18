# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Mingyan Zhao>
<Math 345>
<10/28/2018>
"""
import numpy as np
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    A = A.astype(float)
    #store the dimension of A
    m, n = A.shape
    #initialize Q,R
    Q = A.copy()
    R = np.zeros((n,n))
    #the process
    for i in range(n):
        R[i,i] = la.norm(Q[:,i])
        #normalize the ith column of Q
        Q[:,i] = Q[:,i]/R[i,i] 
        for j in range(i+1, n):
            R[i,j] = (Q[:,j].T)@(Q[:,i])
            #orthogonalize the jth column of Q
            Q[:,j] = Q[:,j] - R[i,j]*Q[:,i]
    return Q, R


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    A = A.astype(float)
    
    return abs(np.prod(np.diag(la.qr(A)[1])))
    

# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    A = A.astype(float)
    b = b.astype(float)
    n = A.shape[0]
    
    #computer Q and R
    Q,R = la.qr(A)
    #computer y
    Y = Q.T@b
    
    X = np.zeros(n)
    #back substitution
    for i in reversed(range(n)):
        summation = 0
        
        for j in range(i, n):
            summation += R[i][j]*X[j]/R[i][i]
            
        X[i] = (Y[i]/R[i][i] - summation)
        
    return X

# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    A = A.astype(float)
    #initialize
    m,n = A.shape
    R = A.copy()
    Q = np.eye(m)
    sign = lambda x: 1 if x>= 0 else -1
    for k in range(n):
        u = np.copy(R[k:,k])
        #u0 is first entry of u
        u[0] = u[0] + sign(u[0])*la.norm(u)
        #normalize u
        u = u/la.norm(u)
        #aplly reflections to R nad Q
        R[k:,k:] = R[k:,k:] - 2*np.outer(u,(u.T@R[k:,k:]))
        Q[k:,:] = Q[k:,:] - 2*np.outer(u,(u.T@Q[k:,:]))
    return Q.T, R
        
        

# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    A = A.astype(float)
    
    m,n = A.shape
    H = A.copy()
    Q = np.eye(m)
    sign = lambda x: 1 if x>= 0 else -1
    
    for k in range(n-2):
        u = H[k+1:,k].copy()
        u[0] = u[0] + sign(u[0])*la.norm(u)
        u = u/la.norm(u)
        #apply Qk to H
        H[k+1:,k:] = H[k+1:,k:] - 2*np.outer(u,(u.T@H[k+1:,k:]))
        #apply QKT to H
        H[:,k+1:] = H[:,k+1:] - 2*np.outer((H[:,k+1:]@u), u.T)
        #aplly Qk to Q
        Q[k+1:,:] = Q[k+1:,:] - 2*np.outer(u,(u.T@Q[k+1:,:]))
    
    return H, Q.T 
        
        

        
        
        
        
        
        
        
        
        
        
        