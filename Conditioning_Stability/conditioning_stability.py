# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Mingyan Zhao>
<Math 347>
<01/29/2019>
"""

import numpy as np
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    #calculate the single values
    sigma = la.svdvals(A)
    max_ = max(sigma)
    min_ = min(sigma)
    #calculate the condition number
    #If the smallest singular value is 0, return ∞ (np.inf).
    if min_ == 0:
        return np.inf
    else:
        return max_/ min_

# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    new_r = np.array([])
    #the absolute condition number
    b = []
    #the relative condition number
    c = []
    for j in range(100):
        r = np.random.normal(1,1e-10, size =21)
        new_coeffs = w_coeffs*r
        new_roots = np.roots(np.poly1d(new_coeffs))
        new_r = np.concatenate([new_r,new_roots])

        # Sort the roots to ensure that they are in the same order.
        w_roots = np.sort(w_roots)
        new_roots = np.sort(new_roots)
        # Estimate the absolute condition number in the infinity norm.
        k = la.norm(new_roots - w_roots, np.inf) / la.norm(r, np.inf)
        b.append(k)
        # Estimate the relative condition number in the infinity norm.
        c.append(k * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf))

    plt.plot(new_r.real, new_r.imag, '.', markersize = 1, label = "Perturbed")
    plt.plot(w_roots.real, w_roots.imag, 'b.', markersize = 10, label = "Original")
    plt.legend()
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.title("Wilkinson polynomial")
    plt.show()
    return np.mean(b), np.mean(c)



# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    #initialize the matrix H
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags
    A_ = A + H
    #calcualte the eigenvalues
    Aval = la.eigvals(A)
    A_val = la.eigvals(A_)
    #claculate the absolute and relative condition number
    k_ = la.norm(Aval-A_val,2)/la.norm(H,2)
    k = la.norm(A,2)*k_/la.norm(Aval,2)
    return k_, k

# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    x_ = np.linspace(domain[0], domain[1], res)
    # Real parts.
    y_ = np.linspace(domain[2], domain[3], res)
    # Imaginary parts.
    r = []
    #check the condition number for each coordinate
    for x in x_:
        for y in y_:
            A = np.array([[1,x],[y,1]])
            r.append(eig_cond(A)[1])

    r = np.array(r)
    r.resize(res,res)
    #plot the basins of attraction
    plt.pcolormesh(x_,y_,r, cmap="gray_r")
    plt.colorbar()
    plt.title("conditioning of the eigenvalue problem")
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.show()


# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
        #load the data
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n+1)

    #solve the normal equation
    x1 = la.inv(A.T@A)@A.T@yk
    #solve the system
    q, r = la.qr(A, mode = 'economic')
    x2 = la.solve_triangular(r, q.T@yk)
    #plot different coefficients
    plt.scatter(xk,yk, marker = 7, alpha = 0.5, )
    plt.plot(xk,np.polyval(x1, xk), label = "Normal Equations")
    plt.plot(xk,np.polyval(x2, xk), label = "QR Solver")
    plt.legend()
    plt.show()
    return la.norm(A@x1-yk,2),la.norm(A@x2-yk,2)


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """

    x = sy.symbols("x")
    domain = np.linspace(5,50,10, dtype=int)
    err = []
    for n in domain:
        #calculate the intergral
        f = (x**n)*sy.exp(x-1)
        I1 = float(sy.integrate(f, (x,0,1)))
        #calculate through the formula
        I2 = (-1)**n*(sy.subfactorial(n)-sy.factorial(n)/np.exp(1))
        #calculate the forward erro
        err.append(np.abs(I1-I2))

    #plot the forward error
    plt.semilogy(domain, err)
    plt.xlabel("n")
    plt.ylabel("Forward Error")
    plt.title("relative forward error")

    plt.show()
