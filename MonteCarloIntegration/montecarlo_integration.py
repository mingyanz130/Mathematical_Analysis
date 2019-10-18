# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Mingyan Zhao>
<Math 321>
<02/12/2019>
"""
import numpy as np
from scipy import linalg as la
from scipy import stats
from matplotlib import pyplot as plt

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    #draw points
    points = np.random.uniform(-1,1, (n,N))
    #find the points in the unit ball
    lengths = la.norm(points, axis=0)
    num_within = np.count_nonzero(lengths < 1)
    #return the estimation
    return 2**n * (num_within / N)

# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    #draw points
    points = np.random.uniform(a,b,N)
    #calculate the volume
    sum = 0
    for i in range(N):
        sum += f(points[i])
    #return the answer
    return (b-a)/N*sum


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    #initialize the list
    maxs = np.array(maxs, dtype = "float")
    mins = np.array(mins, dtype = "float")
    n =len(mins)
    #shift the points
    points = np.random.uniform(0,1,(N,n))*(maxs-mins)+ mins
    #calculate the volume
    V = np.product(maxs-mins)
    #return the answer
    return float(V*sum([f(point) for point in points])/N)
    


# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    #initialize the function and range
    f = lambda x: 1/((2*np.pi)**2)*np.exp(-x.T@x/2)
    mins = [-3/2,0,0,0]
    maxs = [3/4,1,1/2,1]

    #calculate the exact answer
    means, cov = np.zeros(4), np.eye(4)
    answer2 = stats.mvn.mvnun(mins, maxs, means, cov)[0]
    #for different value of n
    domain = np.floor(np.logspace(1,5,20))
    error = []
    #calculate the error
    for n in domain:
        n = int(n)
        answer1 = mc_integrate(f, mins, maxs, N = n)
        error.append(abs((answer2-answer1)/answer2))

    #plot the error and 1/sqrt(N)
    plt.loglog(domain, error, label = "Relative Error")
    plt.loglog(domain, 1/np.sqrt(domain), label = "1/ sqrt(N)")
    plt.legend()
    plt.title("Monte Carlo Convergence")
    plt.show()
