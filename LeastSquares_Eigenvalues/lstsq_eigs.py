# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Mingyan Zhao>
<Math 321>
<09/22/2018>
"""

# (Optional) Import functions from your QR Decomposition lab.
#import sys
#sys.path.insert(1, "../QR_Decomposition")
#from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    A = A.astype(float)
    b = b.astype(float)
    #reduced QR decomposition
    Q, R = la.qr(A, mode="economic")
    #solve for x^
    return la.solve_triangular(R, Q.T@b)
# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    #initialize A and b
    data = np.load("housing.npy")
    n = len(data[:,0])
    A = np.column_stack((data[:,0], np.ones((n,1))))
    b = data[:,1]
    #solve for the leastest square solution
    x_ = least_squares(A, b)

    #plot the scatter plot and the regression line
    x = np.linspace(0, 17, 100)
    plt.scatter(data[:,0], data[:,1], cmap='viridis')
    plt.plot(x, x_[0]*x+x_[1])
    plt.xlabel("Year")
    plt.ylabel("Price index")
    plt.show()


# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    #store the data
    data = np.load("housing.npy")
    n = len(data[:,0])
    b = data[:,1]

    #initialize matrices for degree of 3, 6, 9, and 12
    A3= np.vander(data[:,0],4)
    A6= np.vander(data[:,0],7)
    A9= np.vander(data[:,0],10)
    A12= np.vander(data[:,0],13)

    #solve for x
    x3=la.lstsq(A3,b)[0]
    x6=la.lstsq(A6,b)[0]
    x9=la.lstsq(A9,b)[0]
    x12=la.lstsq(A12,b)[0]

    #initialize the polynomial
    f3 = np.poly1d(x3)
    f6 = np.poly1d(x6)
    f9 = np.poly1d(x9)
    f12 = np.poly1d(x12)

    x = np.linspace(0, 16, 100)

    #graph for different degrees
    ax1 = plt.subplot(221)
    plt.scatter(data[:,0], data[:,1], cmap='viridis')
    plt.plot(x, f3(x))
    plt.xlabel("Year")
    plt.ylabel("Price index")
    plt.title("polynomial of degree 3")

    ax2 = plt.subplot(222)
    plt.scatter(data[:,0], data[:,1], cmap='viridis')
    plt.plot(x, f6(x))
    plt.xlabel("Year")
    plt.ylabel("Price index")
    plt.title("polynomial of degree 6")

    ax3 = plt.subplot(223)
    plt.scatter(data[:,0], data[:,1], cmap='viridis')
    plt.plot(x, f9(x))
    plt.xlabel("Year")
    plt.ylabel("Price index")
    plt.title("polynomial of degree 9")

    ax4 = plt.subplot(224)
    plt.scatter(data[:,0], data[:,1], cmap='viridis')
    plt.plot(x, f12(x))
    plt.xlabel("Year")
    plt.ylabel("Price index")
    plt.title("polynomial of degree 12")


    plt.show()

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")




# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    #initialize data, A and b
    xk, yk = np.load("ellipse.npy").T
    A = np.column_stack((xk**2, xk, xk*yk, yk, yk**2))
    b = np.ones_like(xk)
    #calculate for coeffcients
    a, b_, c, d, e = la.lstsq(A,b)[0]
    #plot the data and the ellipse
    plot_ellipse(a, b_, c, d, e)
    plt.plot(xk, yk, 'k*')
    plt.axis("equal")
    plt.show()

# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    #intialize X
    m, n = np.shape(A)
    x0 = np.random.random((n,1))
    #normalize x0
    x0 = x0/la.norm(x0)
    for k in range(1,N):
        x1 = A@x0
        x1 = x1/la.norm(x1)
        # test if it qualifies

        if la.norm(x1-x0)< tol:
            return x1.T@A@x1, x1
        else:
            x0 = x1

    return x1.T@A@x1, x1

# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """


    m,n = np.shape(A)
    #put A in upper Hessenberg form
    S = la.hessenberg(A)
    for k in range(N):
        #get the QR decomposition of Ak
        Q, R = la.qr(S)
        #Recombine Rk and Qk into Ak+1
        S = R@Q

    #initialize an empty list of eigenvalues
    eigs = []
    i = 0
    while i < n:
        if i == n-1 or S[i+1,i]<tol:
            eigs.append(S[i,i])
        else:
            eig1 = ((S[i,i]+S[i+1,i+1])+cmath.sqrt((S[i,i]+S[i+1,i+1])**2-4*(S[i,i]*S[i+1,i+1]-S[i,i+1]*S[i+1,i])))/2
            eig2 = ((S[i,i]+S[i+1,i+1])-cmath.sqrt((S[i,i]+S[i+1,i+1])**2-4*(S[i,i]*S[i+1,i+1]-S[i,i+1]*S[i+1,i])))/2
            eigs.append(eig1)
            eigs.append(eig2)
            i += 1
        #move to next Si
        i += 1

    return eigs
