# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File.
<Mingyan Zhao>
<Math 321>
<11/26/2018>
"""
import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from imageio import imread



# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    eval, evec = la.eig(A.conj().T@A) #Calculate the eigenvalues and eigenvectors of AH A
    sval = np.sqrt(eval) #Calculate the singular values of A.
    index = np.argsort(sval)[::-1] #Sort the singular values from greatest to least.

    sval = np.array([sval[i] for i in index])
    evec = np.array([evec[i] for i in index]) #Sort the eigenvectors the same way as in the previous step.

    #Count the number of nonzero singular values (the rank of A).
    rank = 0
    for j in sval:
        if j > tol:
            rank += 1
    #Keep only the positive singular values.
    sval1 = sval[:rank]
    #Keep only the corresponding eigenvectors.
    evec1 = evec[:,:rank]
    #Construct U with array broadcasting.
    U1 = A@evec1/sval1

    return U1, sval1, evec1.conj().T

# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    #initialize the unit circle
    x = np.linspace(0, 2*np.pi, 200)
    S1 = np.cos(x).reshape(1,200)
    S2 = np.sin(x).reshape(1,200)
    S = np.vstack((S1,S2))
    # initialize the unit vector
    E = np.array([[1,0,0],[0,0,1]])
    #Compute the full SVD
    u, s, vh = la.svd(A)
    s = np.diag(s)

    #Plot four subplots to demonstrate each step of the transformation,
    #plotting S and E, V HS and V HE, ΣV HS and ΣV HE, then UΣV HS and UΣV HE.
    plt.subplot(221)
    plt.plot(S[0],S[1])
    plt.plot(E[0], E[1])
    plt.title("S and E")
    plt.axis("equal")

    plt.subplot(222)
    plt3 = vh@S
    plt4 = vh@E
    plt.plot(plt3[0],plt3[1])
    plt.plot(plt4[0],plt4[1])
    plt.title("V.H@S and V.H@E")
    plt.axis("equal")

    plt.subplot(223)
    plt5 = s@vh@S
    plt6 = s@vh@E
    plt.plot(plt5[0],plt5[1])
    plt.plot(plt6[0],plt6[1])
    plt.title("s@V.H@S and s@V.H@E")
    plt.axis("equal")

    plt.subplot(224)
    plt7 = u@s@vh@S
    plt8 = u@s@vh@E
    plt.plot(plt7[0],plt7[1])
    plt.plot(plt8[0],plt8[1])
    plt.title("U@s@V.H@S and U@s@V.H@E")
    plt.axis("equal")
    plt.show()


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    if np.linalg.matrix_rank(A)<s:
        raise ValueError("s is greater than the number of nonzero singular values of A")
    #compute the compact SVD of A
    U,S,Vh = la.svd(A, full_matrices=False)
    #stripping off the appropriate columns and entries from U1, Σ1, and V1
    S1 = S[:s]
    Vh1 = Vh[:s,:]
    U1 = U[:,:s]
    As = U1@np.diag(S1)@Vh1
    #Return the best rank s approximation As
    #return the number of entries required to store the truncated form U􏰘Σ􏰘V􏰘H
    return As, U1.size + S1.size + Vh1.size

# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    #Compute the compact SVD of A
    U,S,Vh = la.svd(A, full_matrices=False)
    #compute the lowest rank approximation As of A with 2-norm error less than ε.
    indices = np.where(S < err)[0]
    #If ε is less than or equal to the smallest singular value of A, raise a ValueError
    if indices.size == 0:
        raise ValueError("ε is less than or equal to the smallest singular value of A")
    s = indices[np.argmax(S)]
    return svd_approx(A, s)





# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    # Send the RGB values to the interval (0,1).
    image = imread(filename) / 255

    #handle both grayscale and color images.
    if len(image.shape) == 3:
        #Calculate the low- rank approximations Rs, Gs, and Bs separately, then
        #put them together in a new 3-dimensional array of the same shape as the original image.
        red_layer = image[:,:,0]
        red_layer, size1 = svd_approx(red_layer, s)
        green_layer = image[:,:,1]
        green_layer, size2 = svd_approx(green_layer, s)
        blue_layer = image[:,:,2]
        blue_layer, size3 = svd_approx(blue_layer, s)
        size = size1 + size2 + size3
        r_image = np.dstack((red_layer, green_layer, blue_layer))
        #Set any values outside of the interval [0, 1] to the closer of the two boundary values.
        r_image = np.clip(r_image,0,1)
        plt.suptitle("It saves " + str(image.size - size) + " entries")
        plt.subplot(121)
        plt.imshow(image)
        plt.title("original")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(r_image)
        plt.axis("off")
        plt.title("compressed")
        plt.show()
    else:
        #compute the best rank-s approximation of the image.
        r_image, size = svd_approx(image, s)
        r_image = np.clip(r_image,0,1)
        plt.suptitle("It saves " + str(image.size - size) + " entries")

        plt.subplot(121)
        plt.imshow(image, cmap="gray")
        plt.title("original")
        plt.axis("off")

        plt.subplot(122)
        plt.imshow(r_image, cmap="gray")
        plt.title("compressed")
        plt.axis("off")

        plt.show()
