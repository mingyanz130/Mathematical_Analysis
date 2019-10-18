# drazin.py
"""Volume 1: The Drazin Inverse.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy import sparse

# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    
    #check first condition
    if np.allclose(A@Ad,Ad@A) == False:
        return False
    #check second condition
    if np.allclose(np.linalg.matrix_power(A, k+1)@Ad,np.linalg.matrix_power(A, k)) == False:
        return False
    #check third condition
    if np.allclose(Ad@A@Ad, Ad) == False:
        return False

    return True





# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    n,k = A.shape
    f = lambda x: abs(x) > tol
    g = lambda x: abs(x) <= tol
    #Sort the Schur decomposition.
    Q1, S, k1 = la.schur(A, sort=f)
    Q2, T, k2 = la.schur(A, sort=g)

    #Concatenate part of S and T column-wise.
    U = np.hstack((S[:,:k1],T[:,:(n-k1)]))
    U_ = la.inv(U)
    V = U_@A@U
    #The n × n zero matrix as floats, not ints.
    Z = np.zeros((n,n), dtype=float)
    if k1 != 0:
        M_ = la.inv(V[:k1,:k1])
        Z[:k1,:k1] = M_
    return U@Z@U_


# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    n = len(A)
    #initilize the R and Laplacian matrix
    R = np.zeros((n,n))
    I = np.eye(n)
    L= sparse.csgraph.laplacian(A)
    #update each entry with the formula
    for i in range(n):
        for j in range(n):
            if i != j:
                Lj = L.copy()
                Lj[j,:] = I[j,:]
                LJ = drazin_inverse(Lj)
                R[i,j] = LJ[i,i]
    
    return R
                


# Problems 4 and 5
class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.

        Parameters:
            filename (str): The name of a file containing graph data.
        """
        with open(filename) as file:
            data = file.read().split("\n")
        
        pair = []
        names = set()
        for line in data[:-1]:
            line = line.split(",")
            names.add(line[0])
            names.add(line[1])
            pair.append(line)
        names = sorted(list(names))
        
        n = len(names)
        M = np.zeros((n,n))
        for k in pair:
            i, j  = names.index(k[0]),names.index(k[1])
            M[i,j] = 1
            M[j,i] = 1
        
        self.names = names
        self.M = M
        self.R = effective_resistance(M)
 

    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.

        Parameters:
            node (str): The name of a node in the network.

        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.

        Raises:
            ValueError: If node is not in the graph.
        """
        #find where there is no connection between nodes
        mask = self.M == 1
        R = self.R.copy()
        R[mask] = np.inf
        np.fill_diagonal(R,np.inf)
        #check if the node is in the list
        if node is not None and node not in self.names:
            raise ValueError("The name is not in the list")
        #check if no node is provided, return the names of the next nodes
        elif node == None:
            i,j =  np.where(R==np.min(R))
            return self.names[i[0]],self.names[j[0]]
        #return the node can be conncected to the given node
        else:
            #take the row out and find the index of the minimum
            index = self.names.index(node)
            row = R[:,index]
            k =np.argmin(row)
            return self.names[k]
            
            
    def add_link(self, node1, node2):
        """

        Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.
        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        #check if given nodes are in the list
        if node1 in self.names and node2 in self.names:
            #get the node and add connection
            i,j = self.names.index(node1), self.names.index(node2)
            self.M[i,j] = 1
            self.M[j,i] = 1
            self.R = effective_resistance(self.M)
            
        else:
            raise ValueError("The name is not in the list.")
            

if __name__ == "__main__":
    A = np.array([[1,3,0,0],[0,1,3,0],[0,0,1,3],[0,0,0,0]])
    Ad = np.array([[1,-3,9,81],[0,1,-3,-18],[0,0,1,3],[0,0,0,0]])
    #print(is_drazin(A, Ad, k=1))
    #print(drazin_inverse(A, tol=1e-4))
    B = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]])
    C = np.array([[0,2],[2,0]])
    #print(effective_resistance(B))
    a = LinkPredictor()
    print(a.predict_link("Emily"))
    print(a.predict_link())
    
    