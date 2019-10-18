# solutions.py
# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name>
<Class>
<Date>
"""

import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from scipy import sparse

# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot = False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    #initilize the matirices
    n = A.shape[0]
    D_ = np.diag(1/A.diagonal())
    x0 = np.zeros(n)
    err = []
    
    for i in range(maxiter):
        #updatex_i and err
        x1 = x0 + D_@(b-A@x0)
        err.append(la.norm(A@x1-b,np.inf))
        #check the condition
        if la.norm(x1-x0,np.inf) < tol:   
            if plot:
                #plot the err
                domain = np.arange(len(err))
                plt.semilogy(domain, err)
                plt.title("Convergence of Jacobi Method")
                plt.xlabel("Iteration")
                plt.ylabel("Absolute Error of Approximation")
                plt.show()
                return x1
            else:
                return x1
        x0 = x1.copy()
    
    if plot:
        #plot the err
        domain = np.arange(len(err))
        plt.semilogy(domain, err)
        plt.title("Convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()
        return x1
    else:
        return x1
                    
       
    
    
    
# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #initilize the matrices
    n = A.shape[0]
    x0 = np.zeros(n)
    x1 = np.zeros(n)
    err = []
    while True:
        for k in range(maxiter):
            for i in range(n):
                #update the x_i
                x1[i] = x0[i] + (b[i]-A[i,:]@x0)/A[i,i]
            err.append(la.norm(A@x0-b,np.inf))
            #check the condition
            if la.norm(x1-x0,np.inf) < tol:   
                if plot:
                    #plot the graph
                    domain = np.arange(len(err))
                    plt.semilogy(domain, err)
                    plt.title("Convergence of Jacobi Method")
                    plt.xlabel("Iteration")
                    plt.ylabel("Absolute Error of Approximation")
                    plt.show()
                    return x1
                else:
                    return x1
            x0 = x1.copy()
        if plot:
            #plot the graph
            domain = np.arange(len(err))
            plt.semilogy(domain, err)
            plt.title("Convergence of Gauss-Seidel algorithm")
            plt.xlabel("Iteration")
            plt.ylabel("Absolute Error of Approximation")
            plt.show()
            return x1
        else:
            return x1
# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    #initilize the matrices
    n = A.shape[0]
    x0 = np.zeros(n)
    x1 = np.zeros(n)
    
    iter = 0 
    while iter < maxiter:
        for i in range(n): 
            # Get the indices of where the i-th row of A starts and ends if the
            # nonzero entries of A were flattened.
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            # Multiply only the nonzero elements of the i-th row of A with the
            # corresponding elements of x.
            Aix = A.data[rowstart:rowend] @ x0[A.indices[rowstart:rowend]]
            #update the x_i
            x0[i] = x0[i] + 1/(A[i,i])*(b[i]-Aix) 

        iter += 1
        #check the condition
        if la.norm(x0-x1,np.inf) < tol:   
            break
        x1 = np.copy(x0)
        

    return x0
    
# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    #initilize the matrices
    n = A.shape[0]
    x0 = np.zeros(n)
    x1 = np.zeros(n)
    iter = 0
    converged = False
    while iter < maxiter:
        for i in range(n): 
            # Get the indices of where the i-th row of A starts and ends if the
            # nonzero entries of A were flattened.
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            # Multiply only the nonzero elements of the i-th row of A with the
            # corresponding elements of x.
            Aix = A.data[rowstart:rowend] @ x0[A.indices[rowstart:rowend]]
            #update the x_i
            x0[i] = x0[i] + omega/(A[i,i])*(b[i]-Aix) 
            
        iter += 1
        #check the condition
        if la.norm(x1-x0,np.inf) < tol:   
            converged = True
            break
        x1 = np.copy(x0)

    return x0, iter, converged
# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    #initilize the B and A
    B = sparse.diags([1,-4,1],[-1,0,1],shape=(n,n))
    A = sparse.block_diag([B]*n)
    A2 = sparse.diags([1,1],[-n,n],shape=(n**2,n**2))
    A = A+A2
    b = np.zeros(n)
    b[0], b[-1] = -100, -100
    b = np.tile(b,n)
    
    #find the solution
    sol = sor(A, b, omega,tol,maxiter)
    u = sol[0].reshape(n,n)
    if plot == True:
        #plot the heat map
        plt.pcolormesh(u,cmap = "coolwarm")
        plt.title("The heat map")
        plt.axis("off")
        plt.show()
    return sol
    

# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    #set the domain
    domain = np.arange(1,2,0.05)
    y = []
    for i in domain:
        #record the iteration
        k = hot_plate(20, i, tol=1e-2, maxiter=1000, plot=False)[1]
        y.append(k)
    #plot the relation
    plt.plot(domain,y)
    plt.xlabel("the value of ω")
    plt.ylabel("the number of iteration")
    plt.show()
        
    

if __name__ == "__main__":
    
    n=30
    print("hello")
    b = np.random.random(n)
    A = diag_dom(n)
    x = jacobi(A, b,plot=True)
    #x = gauss_seidel(A, b,plot=True)
    
    print(A@x)
    print(b)
    print(np.allclose(A@x,b))
    
    
    x = gauss_seidel(A, b,plot=True)
    print(A@x)
    print(b)
    print(np.allclose(A@x,b))
    

    A = sparse.csr_matrix(diag_dom(5000))
    b = np.random.random(5000)
    
    
    x = gauss_seidel_sparse(A, b,maxiter=100)
    print(A@x)
    print(b)
    print(np.allclose(A@x,b))
    
    x = sor(A, b, 1,maxiter=100)
    x = x[0]
    
    print(A@x)
    print(b)
    print(np.allclose(A@x,b))
    print(hot_plate(50, omega=1.75,tol=1e-2,maxiter =1000,plot=True))
    prob7()
    
    
    