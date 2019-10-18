# python_intro.py
"""Python Essentials: Introduction to Python.
<Mingyan Zhao>
<Math 345>
<09/06/2018>
"""

#Problem 1

#Problem 2
def sphere_volume(r):
    """
    Return the volume of the sphere of radius r
    """
    return ((4/3)*3.14159*(r**3))

#Problem 3
def isolate (a, b, c, d, e):
    print (a, b, c, sep='     ', end =' ')
    print (d, e)

#Problem 4
def first_half(str = ""):
    n = len(str) // 2
    return str[:n]

def backward(str = ""):
    return str[::-1]

#Problem 5
def list_ops():
    list = ["bear", "ant", "cat", "dog"]
    list.append("eagle")
    list[2] = "fox"
    list.pop(1)
    list = sorted(list, reverse=True)
    list[list.index("eagle")] = "hawk"
    list[-1] = list[-1]+"hunter"
    return list

#Problem 6
def pig_latin(word = ""):
    #check if the first letter is a vowel
    if  word[0] in "aeiou":
        return (word + "hay")
    else:
        return (word[1:] + word[0] + "ay")

#Problem 7
def isPalindrome(n):
    s = str(n)
    reverseString = s[::-1]
    return reverseString == s

# returns largest palindrome that is a multiple of two 3 digit numbers
# and returns -1 if no such palindrome exists
def palindrome():
    palindrome = -1

    for i in range (999, 99, -1):
        for j in range (i, 99, -1):

            # if product is palindrome and is greater than last recorded palindrome
            if isPalindrome(i * j) and i * j > palindrome:
                palindrome = i * j
                q = i #first number
                w = j #second number
    return palindrome


#Problem 8
#Input would be the number of terms we want to add
def alt_harmonic(N):
    sum = 1 # when n = 1, set the initial
    #we need first n terms
    for n in range(2, N + 1):
        sum = sum + ((-1)**(n+1))/n
    return sum

if __name__ == "__main__":
    print("Hello, world!")
    print(sphere_volume(4))
    isolate(1,2,3,4,5)
    print(first_half("Mingyan"))
    print(backward("Mingyan"))
    print(list_ops())
    print(pig_latin('alligator'))
    print(palindrome())
    print(alt_harmonic(1000))
