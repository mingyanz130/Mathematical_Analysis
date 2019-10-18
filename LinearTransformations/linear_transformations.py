# linear_transformations.py
"""Volume 1: Linear Transformations.
<Mingyan Zhao>
<Math 345>
<09/25/2018>
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time

def test():
    data = np.load("horse.npy")
    ax1 = plt.subplot(231)
    plt.plot(data[0], data[1], 'k,')
    plt.axis([-1, 1, -1, 1])
    plt.gca().set_aspect("equal")
    
    ax2 = plt.subplot(232)
    B = stretch(data, 1/2, 6/5)
    plt.plot(B[0], B[1], 'k,')
    plt.axis([-1, 1, -1, 1])
    plt.gca().set_aspect("equal")
    
    ax3 = plt.subplot(233)
    B = shear(data, 1/2, 0)
    plt.plot(B[0], B[1], 'k,')
    plt.axis([-1, 1, -1, 1])
    plt.gca().set_aspect("equal")
    
    ax4 = plt.subplot(234)
    B = reflect(data, 0, 1)
    plt.plot(B[0], B[1], 'k,')
    plt.axis([-1, 1, -1, 1])
    plt.gca().set_aspect("equal")
    
    ax4 = plt.subplot(235)
    B = rotate(data, np.pi/2)
    plt.plot(B[0], B[1], 'k,')
    plt.axis([-1, 1, -1, 1])
    plt.gca().set_aspect("equal")
    
    plt.show()
    
# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    #the stretch transformation
    B = np.array([[a, 0],[0, b]])
    B = B@A
    return B
    
def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    #the shear transformation
    B = np.array([[1, a],[b, 1]])
    B = B@A
    return B



def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    #the reflect transformation
    r = (a**2+b**2)
    B = np.array([[(a**2-b**2)/r, 2*a*b/r],[2*a*b/r, (b**2-a**2)/r]])
    B = B@A
    return B


def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    #the rotate transformation
    B = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    B = B@A
    return B



# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    P_e0 = np.array([[x_e],[0]])
    P_m0 = np.array([[x_m],[0]])
        
    t = np.linspace(0, T, 10000) 
    
    P_e = P_e0.copy()
    P_m = P_m0.copy()
    

    
    #calculator the location for moon and earth, then store them in to a list
    #the first row is x-coordinates, second row is y-coordinates
    for i in t:
        
        P_e1 = rotate(P_e0, i * omega_e)
        P_e = np.hstack([P_e, P_e1])
        
        P_m1 = rotate(P_m0-P_e0, i * omega_m) + P_e1
        P_m = np.hstack([P_m, P_m1])
        
        
    #Plot both orbits
    plt.plot(P_e[0], P_e[1], 'b', label = "Earth")
    plt.plot(P_m[0], P_m[1], color ='orange', label = "Moon")
    plt.legend(loc = "lower right")    
    plt.gca().set_aspect("equal")
    
    plt.show()
    


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    # initialize the domain and time for Matrix-Vector Multiplication
    # record the time and store it in a array
    domain1 = 2**np.arange(1,9)
    times1 = []
    for n in domain1:
        start1 = time.time()
        matrix_vector_product(random_matrix(n),random_vector(n))
        times1.append(time.time() -start1)
    
    
    # initialize the domain and time for Matrix-Matrix Multiplication
    # record the time and store it in a array
    domain2 = 2**np.arange(1,9)
    times2 = []
    for n in domain2:
        start2 = time.time()
        matrix_matrix_product(random_matrix(n),random_matrix(n))
        times2.append(time.time() -start2)
      
        
    #plot both graph    
    a= plt.subplot(121)
    a.plot(domain1, times1, 'g.-', linewidth = 2, markersize = 15)
    plt.xlabel("n", fontsize = 14)
    plt.ylabel("seconds", fontsize = 14)
    plt.title("Matrix-Vector Multiplication", fontsize = 18)
    
    b= plt.subplot(122)
    b.plot(domain2, times2, 'b.-', linewidth = 2, markersize = 15)
    plt.xlabel("n", fontsize = 14)
    plt.title("Matrix-Matrix Multiplication", fontsize = 18)
    plt.show()

# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    # initialize the domain and time for Matrix-Matrix Multiplication and Matrix-Vector
    # record the time and store it in a array
    
    domain = 2**np.arange(1,10)
    
    times1 = []
    times2 = []
    times3 = []
    times4 = []
    
    for n in domain:
        start1 = time.time()
        matrix_vector_product(random_matrix(n),random_vector(n))
        times1.append(time.time() -start1)
        
        start2 = time.time()
        matrix_matrix_product(random_matrix(n),random_matrix(n))
        times2.append(time.time() -start2)
        
        start3 = time.time()
        np.dot(random_matrix(n),random_vector(n))
        times3.append(time.time() -start3)
        
        start4 = time.time()
        np.dot(random_matrix(n),random_matrix(n))
        times4.append(time.time() -start4)
        
        
    # plot all the graphs on two subpolts    
    a= plt.subplot(121)
    
    a.plot(domain, times1, 'g.-', linewidth = 2, markersize = 15, label = "Matrix-Vector")
    a.plot(domain, times2, 'b.-', linewidth = 2, markersize = 15, label = "Matrix-Matrix")
    a.plot(domain, times3, 'c.-', linewidth = 2, markersize = 15, label = "Matrix-Vector dot")
    a.plot(domain, times4, 'y.-', linewidth = 2, markersize = 15, label = "Matrix-Matrix dot")
    plt.legend(loc = "upper left")
    
    
    b= plt.subplot(122)

    b.loglog(domain, times1, 'g.-', basex = 2, basey = 2)
    b.loglog(domain, times2, 'b.-', basex = 2, basey = 2)
    b.loglog(domain, times3, 'c.-', basex = 2, basey = 2)
    b.loglog(domain, times4, 'y.-', basex = 2, basey = 2)

    b.loglog(domain, times1, 'g.-', basex = 2, basey = 2, lw =2)
    b.loglog(domain, times2, 'b.-', basex = 2, basey = 2, lw =2)
    b.loglog(domain, times3, 'c.-', basex = 2, basey = 2, lw =2)
    b.loglog(domain, times4, 'y.-', basex = 2, basey = 2, lw =2)
    
    plt.show()
