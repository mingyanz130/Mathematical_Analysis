# image_segmentation.py
"""Volume 1: Image Segmentation.
<Mingyan Zhao>
<Math 345>
<11/10/2018>
"""

import numpy as np
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg
from imageio import imread
from matplotlib import pyplot as plt


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    #calculate D by creating a diagonal matrix with the column sum of A
    D = np.diag(A.sum(axis=0))
    return D - A

# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    #calculate L and eigenvalues of L
    L = laplacian(A)
    eigs, eigvec = la.eig(L)
    num_cc = 0
    #check zeros
    for i in eigs:
        if i < tol:
            num_cc += 1
        #fins the algebraic connectivity of G
    sec_small = sorted(eigs)[1]
    return num_cc, sec_small

# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        #read the image, scale it and save it
        image = imread(filename)
        self.image = image
        self.scaled = image / 255
        #check if it is in color or grayscale
        if self.scaled.shape[-1] == 3:
            #compute its brightess matrix by averaging the RGB values at each pixel
            self.brightness = self.scaled.mean(axis = 2)
            self.flat_brightness = np.ravel(self.brightness)
        else:
            self.flat_brightness = np.ravel(self.scaled)

    # Problem 3
    def show_original(self):
        """Display the original image."""
        #check if it is in color or grayscale
        if self.scaled.shape[-1] == 3:
            plt.imshow(self.scaled)
            plt.axis("off")
            plt.show()
        else:
            plt.imshow(self.scaled, cmap="gray")
            plt.axis("off")
            plt.show()


    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        #initialize A as a sparse matrix and D as a vector
        m = self.scaled.shape[0]
        n = self.scaled.shape[1]
        A = sparse.lil_matrix((m*n,m*n))
        D = np.ones((m*n,1))
        #fill in nonzero elements of A one row at a time
        for i in range(m*n):
            indices, distance = get_neighbors(i, r, m, n)
            for j in range(len(indices)):
                #find the set of all vertices that satisfy the conditions
                if distance[j] < r:
                    A[i, indices[j]] = np.exp(-abs(self.flat_brightness[i]-self.flat_brightness[indices[j]])/sigma_B2-distance[j]/sigma_X2)
                else:
                    A[i, indices[j]] = 0
        #convert A to a csc matrix
        A = A.tocsc()
        #update D as column sum of A
        D = np.array(A.sum(axis=0))[0]
        return A, D



    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        m = self.scaled.shape[0]
        n = self.scaled.shape[1]
        #compute the Laplacian of A
        L = sparse.csgraph.laplacian(A)
        #compute D^(1/2)
        D1 = sparse.diags(1/np.sqrt(D))
        #compute the eigenvector corresponding to the second-smallest eigenvale of D^(1/2)@L@D^(1/2)
        G = D1@L@D1
        #save only the second smallest eigenvector and reshape
        eigval, eigvec = linalg.eigsh(G, which = "SM", k = 2)
        x = eigvec[:,1].reshape(m,n)
        #construct the boolean mask
        mask = x > 0
        return mask


    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        #initialize the matrix with other functions
        A, D = self.adjacency(r,sigma_B, sigma_X)
        x1 = self.cut(A,D)
        #check if it in color or grayscale, plot the original image, the postive segment, and the negtive segment
        if self.scaled.shape[-1] == 3:
            #original
            plt.subplot(131)
            plt.imshow(self.scaled)
            plt.axis("off")
            plt.title("original")
            #positive
            plt.subplot(132)
            plt.imshow(self.scaled*np.dstack((x1,x1,x1)))
            plt.axis("off")
            plt.title("postive")
            #negative
            plt.subplot(133)
            plt.imshow(self.scaled*np.dstack((~x1,~x1,~x1)))
            plt.axis("off")
            plt.title("negative")
            plt.show()

        else:
            #original
            plt.subplot(131)
            plt.imshow(self.scaled, cmap="gray")
            plt.axis("off")
            plt.title("original")
            #positive
            plt.subplot(132)
            plt.imshow(self.scaled*x1, cmap="gray")
            plt.axis("off")
            plt.title("postive")
            #negative
            plt.subplot(133)
            plt.imshow(self.scaled*~x1, cmap="gray")
            plt.axis("off")
            plt.title("negative")
            plt.show()
"""
if __name__ == '__main__':
     ImageSegmenter("dream_gray.png").segment()
     ImageSegmenter("dream.png").segment()
     ImageSegmenter("monument_gray.png").segment()
     ImageSegmenter("monument.png").segment()
"""
