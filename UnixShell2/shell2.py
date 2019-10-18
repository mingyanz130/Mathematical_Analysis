# shell2.py

import os
from glob import glob
import subprocess
import numpy as np

# Problem 5
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.
    """
    #find all files
    files = glob("**/"+file_pattern, recursive=True)
    target = []
    for file in files:
        #read the files
        with open(file, "r") as filename:
            #check the target_string
            if target_string in filename.read():
                #store the file
                target.append(file)
    return target



# Problem 6
def largest_files(n):
    """Return a list of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    #find all files
    files = glob("**/*.*", recursive=True)
    size = []
    for i in files:
        if os.path.isfile(i):
            #find the size
            size.append(os.path.getsize(i))

    #sort the file by size
    order = np.argsort(size)
    files = np.array(files)
    return list(files[order][::-1][:n])
