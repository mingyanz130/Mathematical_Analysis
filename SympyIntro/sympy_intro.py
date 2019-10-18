# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
<Mingyan Zhao>
<Math 346>
<01/15/2019>
"""
import sympy as sy
import numpy as np
import matplotlib.pyplot as plt


# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    x, y = sy.symbols('x,y')
    #return the function with sympy symbols
    return sy.Rational(2,5)*sy.exp(x**2-y)*sy.cosh(x+y)+sy.Rational(3,7)*sy.log(x*y+1)

# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    x, j, i = sy.symbols("x j i")
    #return the simplified function
    return sy.simplify(sy.product(sy.summation(j*(sy.sin(x)+sy.cos(x)),(j,i,5)),(i,1,5)))

# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    #initialize the function
    x, y, n = sy.symbols("x y n")
    func = sy.summation(x**n/sy.factorial(n),(n,0,N))
    #sub the x with -y^2
    new_func = func.subs(x, -y**2)
    #graph the function and Maclaurin series
    domain = np.linspace(-2,2,100)
    f = sy.lambdify(y, new_func, "numpy")
    f_ = lambda x: np.exp(-x**2)
    plt.plot(domain, f(domain),label="series")
    plt.plot(domain,f_(domain), label = "original")
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    #construct the function
    x, y, th, r = sy.symbols("x y th r")
    func = 1 - ((x**2+y**2)**sy.Rational(7,2)+18*x**5*y-60*x**3*y**3+18*x*y**5)/((x**2+y**2)**3)
    #change the function into polar coordinates by subbing
    newfunc = func.subs({x:r*sy.cos(th),y:r*sy.sin(th)})
    newfunc = sy.simplify(newfunc)
    #solve the equation and simplify it
    solution = sy.solve(newfunc,r)
    r = sy.lambdify(th, solution[1],"numpy")
    #graph the function
    domain = np.linspace(0,2*np.pi,100)
    plt.plot(r(domain)*np.cos(domain),r(domain)*np.sin(domain))
    plt.show()

# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    #initialize matrices A and I
    x, y, z, lam = sy.symbols('x y z lam')
    A = sy.Matrix([ [x-y, x, 0],
                    [x, x-y, x],
                    [0, x, x-y] ])
    I = sy.Matrix([ [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1] ])
    #solve for the null space of A-lamda*I
    solution = sy.solve(sy.det(A-lam*I),lam)
    #assign each eigenvectors to each eigenvalues
    eigval = {}
    for l in solution:
        eigval[l] = (A-l*I).nullspace()
    return eigval


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    #initialize the function and calculate the first and second derivatives
    domain = np.linspace(-5,5,100)
    x = sy.symbols("x")
    p = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    dp = sy.diff(p, x)
    d2p = sy.diff(dp, x)
    x0 = sy.solve(dp, x)
    #find the x such that f'(x) = 0 and then check the second derivative
    func = sy.lambdify(x, p, "numpy")
    check = sy.lambdify(x, d2p, "numpy")
    #find the local maximum and minimum
    mini = []
    maxi = []
    for k in x0:
        if check(k)>0:
            mini.append(k)
        elif check(k)<0:
            maxi.append(k)
    #plot the function and its local minimum and maximum
    plt.plot(domain, func(domain), label = "p", color = "blue")
    plt.scatter(np.array(mini), func(np.array(mini)), label = "Minimum", color='green' )
    plt.scatter(np.array(maxi), func(np.array(maxi)), label = "Maximum", color='red' )
    plt.title("p and its local minimum and maximum")
    plt.legend()
    plt.show()

    return set(mini), set(maxi)


# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    #initialize the function
    x, y, z, j, k, l, r = sy.symbols('x y z j k l r')
    func = (x**2 + y**2 + z**2)**2
    #calculate the Jacobian matrix
    h = sy.Matrix([[j*sy.sin(l)*sy.cos(k)], [j*sy.sin(l)*sy.sin(k)], [j*sy.cos(l)]])
    J = h.jacobian((j,k,l))
    f = sy.lambdify((x,y,z),func)
    det_ = -sy.det(J).simplify()
    #calculate the function in terms of r by intergrting the volume
    integ = sy.integrate(f(*h).simplify() * det_, (j, 0, r), (k, 0 , 2*sy.pi), (l, 0, sy.pi))

    g = sy.lambdify(r, integ)
    #graph the function in terms of r
    domain = np.linspace(0,3,100)
    plt.plot(domain, g(domain))
    plt.show()
