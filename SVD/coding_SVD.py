import numpy as np

def SVD(A):
    U, S, VT = np.linalg.svd(A, full_matrices=True)
    print("Test U:")
    print( (np.transpose(U) @ U - U @np.transpose(U)))
    print("Test VT:")
    print( (np.transpose(VT) @ VT - VT @np.transpose(VT)))

    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=S[i]
    return U @ D @ VT

X = np.array([ [1.0,-1.0], [1.0,-1.0]])
SVD = SVD(X)
print("Test for SVDDDD")
print(X - SVD)


"""
As you can see from the code, the
vector must be converted into a diagonal matrix.
This may cause a problem as the size of the matrices do not fit the rules
of matrix multiplication, where the number of columns in a
matrix must match the number of rows in the subsequent matrix.
"""
