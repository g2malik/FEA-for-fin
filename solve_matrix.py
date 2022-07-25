import numpy as np
import matplotlib.pyplot as plt
n = 4
A = [[78.6, -0.4, -25, 0], [-1.6,76.6,0,-25], [-25,0, 78.6, -0.4], [0,-25,-1.6, 76.6]]
A = np.array(A)

Su = [5064,5000,2564,2500]
Su = np.array(Su)

T_init = np.zeros((n,1))
T_matrix = np.copy(T_init)
#print(T_matrix)
iter = 0
n_iterations = 10
while iter < n_iterations:

    for i in range(np.shape(A)[0]):
        GS_num = 0
        for j in range(np.shape(A)[1]):
            if A[i,j] != 0 and i!=j:
                GS_num += -A[i,j]*T_matrix[j]
                
        T_matrix[i] = (GS_num + Su[i]) / A[i,i]

    iter +=1