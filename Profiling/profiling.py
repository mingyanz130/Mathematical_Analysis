# profiling.py
"""Python Essentials: Profiling.
<Mingyan Zhao>
<Math 347>
<01/07/2019>
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import math
import numpy as np
from numba import jit
from numba import int64, double
import time
import matplotlib.pyplot as plt


# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""

    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    n = len(data)
    for j in range(n-1):
        m = len(data[n-j-2])
        for k in range(m):
            data[n-j-2][k] += max(data[n-j-1][k],data[n-j-1][k+1])
    return data[0][0]
# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

def primes_fast(N):
    """Compute the first N primes."""
    primes_list = [2]
    current = 3
    while len(primes_list) < N:
        isprime = True
        current_ = int(math.sqrt(current)) + 1
        for i in primes_list:     # Check for nontrivial divisors.
            if i >= current_:
                break
            if current % i == 0:
                isprime = False
                break

        if isprime:
            primes_list.append(current)
        current += 2
    return primes_list

# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    return np.argmin(np.linalg.norm((A - x.reshape(-1,1)),axis = 0))

# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total

def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))

    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter = {i:c+1 for c, i in enumerate(alphabet)}
    total = 0
    for i, name in enumerate(names):
        total += (i+1)*sum([letter[x] for x in name])
    return total

# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    x1 = 1
    x2 = 1
    yield 1
    yield 1
    while True:
        x = x1 + x2
        x1 = x2
        x2 = x
        yield x
def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    for i, x in enumerate(fibonacci()):
        if x >= 10**(N-1):
            return i+1

# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""

    list_ = np.array(list(range(2,N)))
    while len(list_) != 0:
        mask = list_ % list_[0] != 0
        yield list_[0]
        list_ = list_[mask]


# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit(nopython=True, locals=dict(A=double[:,:],m=int64, n=int64,temporary_array=double[:], total=double))
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""

    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product



def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    domain = [2**i for i in range(2,8)]
    t1 = []
    t2 = []
    t3 = []

    for m in domain:
        A = np.random.random((m,m))

        start1 = time.time()
        matrix_power(A, n)
        t1.append(time.time() - start1)

        matrix_power_numba(A, n)
        start2 = time.time()
        matrix_power_numba(A, n)
        t2.append(time.time() - start2)

        start3 = time.time()
        np.linalg.matrix_power(A,n)
        t3.append(time.time() - start3)

    plt.loglog(domain, t1, basex = 2, label= "original")
    plt.loglog(domain, t2, basex = 2, label = "numba")
    plt.loglog(domain, t3, basex = 2, label = "build_in")
    plt.title("Execution time")
    plt.legend()
    plt.show()
