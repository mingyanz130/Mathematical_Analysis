# newtons_method.py
"""Volume 1: Newton's Method.
<Mingyan Zhao>
<Math 347>
<01/29/2019>
"""
import numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import numpy.linalg as la

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    #check the dimension of the function
    check = np.isscalar(x0)
    if check:
        # compute a zero x ̄ of f
        for k in range(maxiter):
            x1 = x0 - alpha*f(x0)/Df(x0)
            #Terminate the algorithm when |xk − xk−1| is less than
            #the stopping tolerance or after iterating the maximum
            #number of allowed times.
            if abs(x1 - x0) < tol:
                return x1, True, k+1
            x0 = x1
    else:
        # compute a zero x ̄ of f
        for k in range(maxiter):
            yk = la.solve(Df(x0), f(x0))
            x1 = x0 - alpha*yk
            #Terminate the algorithm when |xk − xk−1| is less than
            #the stopping tolerance or after iterating the maximum
            #number of allowed times.
            if la.norm(x1 - x0) < tol:
                return x1, True, k+1
            x0 = x1
    return x1, False, maxiter

# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    #initialize the function
    f = lambda r: P1*((1+r)**N1-1)- P2*(1-(1+r)**(-N2))
    #find the derivative
    Df = grad(f)
    #statring point
    r0 = 0.1
    return newton(f, r0, Df)[0]


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    #initialize the domain
    alpha = np.linspace(0,1,200)[1:]
    iter = []
    #calculate the iteration time for each alpha
    for a in alpha:
        N = newton(f, x0, Df, alpha = a)[2]
        iter.append(N)
    #plot the graph
    plt.plot(alpha, iter)
    plt.show()
    return alpha[np.argmin(iter)]


# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    #initialize the function and the searching domain
    rx = np.linspace(-1/4,0,100)
    ry = np.linspace(0,1/4,100)

    f = lambda p: np.array([5*p[0]*p[1]-p[0]*(1 + p[1]), -p[0]*p[1] + (1-p[1])*(1 + p[1])])
    Df = lambda p: np.array([[5*p[1]-(1+p[1]),4*p[0]],[-p[1],-p[0]-2*p[1]]])
    #check the condition for all three points for different alpha value
    for x in rx:
        for y in ry:
            x0 = np.array([x,y])
            val1, bool1, iter1 = newton(f,x0,Df, alpha =1)
            if np.allclose(val1, np.array([0,1])) or np.allclose(val1, np.array([0,-1])):
                val1, bool1, iter1 = newton(f,x0,Df, alpha =.55)
                if np.allclose(val1, np.array([3.75, 0.25])):
                    return x0

# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    x_real = np.linspace(domain[0], domain[1], res)    # Real parts.
    x_imag = np.linspace(domain[2], domain[3], res)    # Imaginary parts.
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    # Combine real and imaginary parts
    X_0 = X_real + 1j*X_imag
    #repeat the procedure for certain times
    for i in range(iters):
        X_1 = X_0 - f(X_0)/Df(X_0)
        X_0 = X_1
    #check the closest point
    y = np.array([np.argmin(abs(x-zeros)) for x in X_1.ravel()])
    y.resize(res,res)
    #plot the basins of attraction
    plt.pcolormesh(X_real, X_imag, y, cmap="brg")
    plt.title("Basins of attraction")
    plt.xlabel("real")
    plt.ylabel("imagine")
    plt.show()
