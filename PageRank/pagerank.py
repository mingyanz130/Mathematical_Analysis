# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name>
<Class>
<Date>
"""
import numpy as np
import numpy.linalg as la
import operator
import networkx as nx
from itertools import combinations

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        A_ = np.array(A,dtype="float")
        n = len(A_)
        #guarantee that node is no longer a sink
        for i in range(n):
            if np.all(A_[:,i]==0):
                A_[:,i] = np.ones(n)
        #normalized
        A_ = A_/np.sum(A_, axis=0)

        self.A = A_
        #initilize the labels
        if labels == None:
            labels = np.arange(n)
        else:
            if len(labels) !=n:
                raise ValueError("the number of labels is not equal to the number of nodes in the graph")
        self.labels = labels
        self.n = n


    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #take the inverse and solve for P
        k = la.inv(np.eye(self.n)-epsilon*self.A)@ ((1-epsilon)/self.n*np.ones(self.n))
        return dict(zip(self.labels,k))
    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #find the eigen vector
        B = epsilon*self.A + (1-epsilon)/self.n*np.ones((self.n,self.n))
        eigval, eig = la.eig(B)
        p = eig[:,0]
        #Normalize
        k = abs(p/la.norm(p,ord=1))
        return dict(zip(self.labels,k))

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        #initilize
        one = np.ones(self.n)
        p1 = one/self.n
        p0 = np.zeros(self.n)
        #iterate untill the condition is met
        i = 0
        while la.norm(p1-p0, ord=1) >= tol and i< maxiter:
            p0 =p1
            p1 = epsilon*self.A@p1+(1-epsilon)/self.n*one
            i += 1
        return dict(zip(self.labels,p1))


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    #get the keys and values
    keys = list(d.keys())
    val = list(d.values())
    #sort the dictionary by value
    order = np.argsort(val)[::-1]
    return [keys[i] for i in order]


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks().

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    #open the file
    with open(filename,"r") as text_file:
        websites = set()
        lines = text_file.read().split("\n")
        #save all possible websites
        for line in lines:
            web = line.split("/")
            for word in web:
                websites.add(word)

        websites = sorted(list(websites))
        n = len(websites)
        #make the dictionary
        dic = {websites[i]:i for i in range(n)}

        #initilize the matrix
        A = np.zeros((n,n))
        for line in lines:
            web = line.split("/")
            l = len(web)
            #add weights for edges
            for k in range(1,l):
                j = dic[web[k]]
                A[j,dic[web[0]]] += 1

        dict = DiGraph(A, dic.keys()).itersolve(epsilon=epsilon)

    return get_ranks(dict)



# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """

    #open file
    team = set()
    with open(filename,"r") as file:
        lines = file.read().split("\n")[1:-1]
        #get all possible teams
        for line in lines:
            row = line.strip().split(",")
            team.add(row[0])
            team.add(row[1])
        team = sorted(team)
        n=len(team)
        #create a dictionary
        dic = {team[i]:i for i in range(n)}
        A= np.zeros((n,n))
        #add weights for each edge
        for line in lines:
            row = line.split(",")
            A[dic[row[0]],dic[row[1]]] += 1

    dict = DiGraph(A, team).itersolve(epsilon=epsilon)
    return get_ranks(dict)


# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    names = set()
    #get the data in proper format
    with open(filename,"r", encoding="utf-8") as file:
        contents = file.readlines()
        contents = [line.strip().split('/') for line in contents]

    DG = nx.DiGraph()
    #add weights for each nodes
    for c in contents:
        #use all different combinations
        for comb in combinations(c[1:],2):
            i,j = comb[0], comb[1]
            #check if there is an edge
            if DG.has_edge(j,i):
                DG[j][i]["weight"] += 1
            else:
                DG.add_edge(j,i,weight=1)
    return get_ranks(nx.pagerank(DG, alpha=epsilon))



if __name__ == "__main__":
    A = np.array([[0,0,0,0],[1.,0,1.,0],[1.,0,0,1.],[1.,0,1.,0]])
    a = DiGraph(A)
    x = a.itersolve()
    y = a.eigensolve()
    z = a.linsolve()
    print(x)
    print(y)
    print(z)
    print(rank_websites(epsilon=0.48)[:20])
