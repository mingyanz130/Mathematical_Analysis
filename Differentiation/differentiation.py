# solutions.py
"""Volume 1: Differentiation.
<Mingyan Zhao>
<Math 347>
<01/22/2019>
"""
import sympy as sy
from matplotlib import pyplot as plt
import numpy as np
from autograd import numpy as anp
from autograd import grad
from autograd import elementwise_grad
from sympy import factorial
import time
# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    #initialize the function and find its derivative
    x = sy.symbols("x")
    f = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    df = sy.lambdify(x, sy.diff(f,x), "numpy")
    #domain = np.linspace(-np.pi, np.pi, 100)
    #ax = plt.gca()
    #ax.spines["bottom"].set_position("zero")
    #plt.plot(domain, df(domain))
    #plt.plot(domain, sy.lambdify(x, f, "numpy")(domain))
    #plt.show()
    return df

# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    #the first order forward difference
    f1 = (f(x+h) - f(x))/h

    return f1

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    f2 = (-3*f(x) + 4*f(x+h) - f(x+2*h))/(2*h)
    return f2

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    f3 = (f(x) - f(x-h))/h
    return f3

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    f4 = (3*f(x) - 4*f(x-h) + f(x-2*h))/(2*h)
    return f4

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    f5 = (f(x+h) - f(x-h))/(2*h)
    return f5

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    f6 = (f(x-2*h) - 8*f(x-h) + 8*f(x+h)-f(x+2*h))/(12*h)
    return f6


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    #initialize the function f
    f = lambda x:(np.sin(x) + 1)**(np.sin(np.cos(x)))
    #for h = 10^-8, 10^-7, ..., 10^-1, 1.
    h = np.logspace(-8, 0, 9)
    df = prob1()
    #compute the exact value of f'(x0)
    df0 = df(x0)
    #get approximate derivative f'(x0)
    #graph the absolute error against h
    plt.loglog(h, abs(df0 - fdq1(f, x0, h)),'-o', label = "Order 1 Forward")
    plt.loglog(h, abs(df0 - fdq2(f, x0, h)),'-o', label = "Order 2 Forward")
    plt.loglog(h, abs(df0 - bdq1(f, x0, h)),'-o', label = "Order 1 Backward")
    plt.loglog(h, abs(df0 - bdq2(f, x0, h)),'-o', label = "Order 2 Backward")
    plt.loglog(h, abs(df0 - cdq2(f, x0, h)),'-o', label = "Order 2 Centered")
    plt.loglog(h, abs(df0 - cdq4(f, x0, h)),'-o', label = "Order 4 Centered")
    plt.legend()
    plt.show()



# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a forward difference quotient for t=7, a backward difference
    quotient for t=14, and a centered difference quotient for t=8,9,...,13.
    Return the values of the speed at each t.
    """
    #load the file
    file = np.load("plane.npy")
    #get time
    t = file[:,0]
    n = len(t)
    #distance a = 500 m
    a = 500
    #data comes from the radar stations at 1 second interval
    h = 1
    #convert α and β to radians
    alpha = np.deg2rad(file[:,1])
    beta = np.deg2rad(file[:,2])
    #compute the coordinates x(t) and y(t)
    x = a*np.tan(beta)/(np.tan(beta)-np.tan(alpha))
    y = a*np.tan(beta)*np.tan(alpha)/(np.tan(beta)-np.tan(alpha))
    #Approximate x′(t) and y′(t) using a forward difference quotient for t = 7
    xderi = [(x[0+1]-x[0])/h]
    yderi = [(y[0+1]-y[0])/h]
    #a centered difference quotient for t = 8, 9, . . . , 13
    for i in range(1, n-1):
        xderi.append((x[i+h]-x[i-h])/(2*h))
        yderi.append((y[i+h]-y[i-h])/(2*h))
    #a backward difference quotient for t = 14
    xderi.append((x[n-1]-x[n-1-h])/h)
    yderi.append((y[n-1]-y[n-1-h])/h)
    xderi = np.array(xderi)
    yderi = np.array(yderi)
    #Return the values of the speed sqrt(􏰘x′(t)2 + y′(t)2) at each t.
    return np.sqrt(xderi**2+yderi**2)

# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    #get the shape
    n = x.shape[0]
    m = len(f(x))
    #initialize the Jacobian matrix
    J = np.zeros((m,n))
    #get the identity matrix
    I = np.eye(n)
    #Approximate the Jacobian matrix of f at x using the second order
    #centered difference quotient
    for j in range(n):
        J[:,j] = (f(x+h*I[:,j])-f(x-h*I[:,j]))/(2*h)
    return J


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    i = sy.symbols("i")
    T0 = 1
    T1 = i
    if n == 0:
        return anp.ones_like(x)
    elif n == 1:
        return x
    #computes Tn
    for k in range(n-1):
        Tn = 2*i*T1-T0
        T0 = T1
        T1 = Tn
    #lambdify the function
    Tn = sy.lambdify(i, Tn)
    return Tn(x)

def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    #initialize the domain
    domain = np.linspace(-1,1,200)
    #get the derivative
    d_cheb = elementwise_grad(cheb_poly)
    #plot each derivative
    for n in range(5):
        plt.plot(domain, d_cheb(domain, n), label = "n = " + str(n))
    plt.legend()
    plt.title("Chebyshev Polynomial")
    plt.show()






# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    #initilize the function
    f = lambda x: (anp.sin(x) + 1)**anp.sin(anp.cos(x))
    #initialize the list
    exact = []
    app1_ = []
    app2_ = []
    err = []
    err1 = []
    err2 = []

    for i in range(N):
        #choose a randome value x0
        x0 = np.random.random()
        #calculate the “exact” value of f′(x)
        #time it
        start1 = time.time()
        df = prob1()
        dff = df(x0)
        exact.append(time.time() - start1)
        err.append(1e-18)

        #using the fourth-order centered difference quotient
        #time it
        start2 = time.time()
        app1 = cdq4(f, x0, h=1e-5)
        app1_.append(time.time() - start2)
        err1.append(abs(dff - app1))

        #using Autograd
        #time it
        start3 = time.time()
        app2 = grad(f)(x0)
        app2_.append(time.time() - start3)
        err2.append(abs(dff - app2))

    #ploting the computation times versus the absolute errors
    plt.loglog(exact, err,"o",alpha = .5, label = "Sympy")
    plt.loglog(app1_, err1,"o",alpha = .5, label = "Difference Quotients")
    plt.loglog(app2_, err2,"o",alpha = .5, label = "Autograd")
    plt.xlabel("Computation Time (seconds)")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.show()
