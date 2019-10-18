# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<Mingyan Zhao>
<Math 345>
<09/14/18>
"""
import numpy as np

def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""
    A = np.array([[3,-1,4],[1,5,-9]])
    B= np.array([[2,6,-5,3],[5,-8,9,7],[9,-3,-2,-3]])
    
    return A @ B

def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    A = np.array([[3,1,4],[1,5,9],[-5,3,1]])
    
    return -A@A@A + 9 * A@A -15 * A


def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.ones((7,7), dtype = np.int)
    #creat a matrix with only ones, take only upper triangle of the matrix
    A = np.triu(A)
    #create a  matrix with only 7 as the entry, then change upper triangle with -1.
    B = np.full((7,7), 5) 
    for j in range (0,7):
        for k in range (0, j + 1):
            B[j][k] = -1
    x = A@B@A
    #change the type
    x = x.astype(np.int64)
    
    return x


def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    B = A.copy()
    #mark all entries that are negtive, then change it to 0
    mask = B < 0
    B[mask] = 0
    
    return B


def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    #define A,B, andC
    A = np.arange(6).reshape((3,2))
    A = A.T
    B = np.full((3,3),3)
    B = np.tril(B)
    C = np.diag([-2,-2,-2])
    I = np.eye(3)
    
    #create rows by stack them herizontally
    row1 = np.hstack((np.zeros((3,3)),A.T, I))
    row2 = np.hstack((A, np.zeros((2,2)), np.zeros((2,3))))
    row3 = np.hstack((B, np.zeros((3,2)), C)) 
    #create the matrix by stack them vertically
    matrix = np.vstack((row1, row2, row3))
    return matrix

def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    #change the type to float
    A = A.astype(np.float64)
    #recalculate each role
    for i in range(0, len(A)):
        A[i] = A[i]/np.sum(A[i])
    
    
    return A


def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    #load the grid
    grid = np.load("grid.npy")
    
    #check the max horizontally       
    max1 = np.max(grid[:,:-3]*grid[:,1:-2]*grid[:,2:-1]*grid[:,3:])
    #vertically
    max2 = np.max(grid[:-3,:]*grid[1:-2,:]*grid[2:-1,:]*grid[3:,:])
    #diagonally(upleft to downright)
    max3 = np.max(grid[:-3,:-3]*grid[1:-2,1:-2]*grid[2:-1,2:-1]*grid[3:,3:])
    #diagonally(downleft tp upright)
    max4 = np.max(grid[3:,:-3]*grid[2:-1,1:-2]*grid[1:-2,2:-1]*grid[:-3,3:])
    #find the max of four
    Max = max(max1, max2, max3, max4)
    
    return Max

