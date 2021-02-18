import numpy as np

a = np.array([[1,2,3,4,5],
     [11,22,33,44,55],
     [111,222,333,444,555]])
b = ["eins", "zwei", "3", "4", "5"]

print([print(j) for i,j in zip(a,b)])

print(a.newaxis)