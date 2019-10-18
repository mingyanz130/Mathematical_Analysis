# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Mingyan Zhao>
<Math 321>
<09/22/2018>
"""
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    A = np.random.normal(size=(n,n))   
    B = A.mean(axis = 1)  
    
    return np.var(B)    
        
    

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    result = []
    for i in range(100, 1100, 100):
        result.append(var_of_means(i))
    plt.plot(result)
    plt.show()


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2*np.pi,2*np.pi, 100)
    # sinx
    plt.plot(x, np.sin(x))
    #cosx
    plt.plot(x, np.cos(x))
    #arctanx
    plt.plot(x, np.arctan(x))
    plt.show()
    
    
    
# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x1 = np.linspace(-2, 1, 100)
    x1 = np.delete(x1, 99)
    
    x2 = np.linspace(1, 6, 100)
    x2 = np.delete(x2, 0)
    
    plt.plot(x1, 1/(x1-1), 'm--', linewidth = 4)
    plt.plot(x2, 1/(x2-1), 'm--', linewidth = 4)
    plt.xlim(-2, 6)
    plt.ylim(-6, 6)
    plt.show()
   
    
    

# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x = np.linspace(0, 2*np.pi, 100)
    plt.axis([0, 2*np.pi, -2, 2])
    #graph of sinx   
    ax1 = plt.subplot(221)
    ax1.plot(x, np.sin(x), 'g-')
    plt.axis([0, 2*np.pi, -2, 2])
    plt.title("sin(x)", fontsize = 18)
    #graph of sin2x
    ax2 = plt.subplot(222)
    ax2.plot(x, np.sin(2*x), 'r--')
    plt.axis([0, 2*np.pi, -2, 2])
    plt.title("sin(2x)", fontsize = 18)
    #graph of 2sinx
    ax3 = plt.subplot(223)
    ax3.plot(x, 2* np.sin(x),'b--')
    plt.axis([0, 2*np.pi, -2, 2])
    plt.title("2sin(x)", fontsize = 18)
    #graph of 2sin2x
    ax4 = plt.subplot(224)
    ax4.plot(x,2* np.sin(2*x), 'm:')
    plt.axis([0, 2*np.pi, -2, 2])
    plt.title("2sin(2x)", fontsize = 18)
    
    plt.suptitle("graphs", fontsize = 20)
    
    
    plt.show()
    

# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    file = np.load("Fars.npy")
    x = file[:, 2]
    y = file[:, 1]
    
    ax1 = plt.subplot(121)
    ax1.plot(y, x, 'k,')
    plt.ylabel("longitudes")
    plt.xlabel("latitudes")
    plt.axis("equal")
    
    z = file[:, 0]
    ax2 = plt.subplot(122)
    ax2.hist(z, bins = 24)
    plt.xlim(0, 23)
    plt.xlabel("hours of the day")
    
    plt.show()
    
   

# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    x = np.linspace(-2*np.pi, 2*np.pi, 100)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = (np.sin(X)*np.sin(Y))/(X*Y)
     #set the limit
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.ylim(-2*np.pi, 2*np.pi)
    
    #the heat map
    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap="viridis")
    plt.colorbar()
    #the contour map
    plt.subplot(122)
    plt.contour(X, Y, Z, 10, cmap="coolwarm")
    plt.colorbar()
    
   
    
    plt.show()
    