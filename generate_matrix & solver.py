import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

file  = open("A_array", "w")
file2 = open("Su_array", "w")
dx_mm = 0.5
dx = dx_mm/1000 #grid spacing (dx = dy)
k = 100
h = 20
T_infin = 30
T_sat = 100
Bio = h*dx/k

# ---------------------------------Code to setup the matrices---------------------------------------#
anx = 0.016/dx + 1
any = 0.001/dx + 1
bnx = 0.004/dx + 1
bny = 0.003/dx
ny = bny + any
print(ny)
n = (any*anx) + (bny*bnx) # total number of nodes

if n - int(n) != 0:
    print("error! number of elements if a float")
else:
    print("number of nodes =", n)
    n = int(n)
    anx = int(anx)
    any = int(any)
    bnx = int(bnx)
    bny = int(bny)

a_start = 0
a_end = (anx * (any-1)) - 1
c_start = a_end + 1
c_end = (anx * any) - 1
b_start = c_end +1
b_end = n - 1
d_start = c_end + 2
d_end = d_start + bnx - 3

def reshape2 (array_1d, fill):
    array_2d = np.full((any+bny, anx), fill) #Fill is the value to fill the rest of the matrix
    matrix1 = (np.reshape(array_1d[:c_end+1], (any, anx))) #slicing doesn't include the last element
    matrix2 = (np.reshape(array_1d[b_start:], (bny, bnx)))
    array_2d[:any, :anx] = matrix1
    array_2d[any:, :bnx] = matrix2

    return array_2d

def convert (one_d_i):
    return (np.where(index_2d == one_d_i))

def where2 (i_array):
    x_array = []
    y_array = []
    
    for val in i_array:
        x_array.append(np.where(index_2d == val)[1])
        y_array.append(np.where(index_2d == val)[0])
    x_array = np.array(x_array)
    y_array = np.array(y_array)
    return x_array, y_array

#initilization of the 1D and 2D Matrix containing the inidexes and functions to convert 1D to 2D indexes for plotting
index_1d = np.arange(0,n,1)
index_2d = reshape2(index_1d, 0.5) #0.5 is an arbritary fill value as long as not integer (initialization)

# Initialization of the matrices for the problem
A = np.zeros((n,n))
Su = np.zeros((n,1))

# corners 1,2,3,4
c1 = anx-1 
c2 = c_end
c3 = c_start + bnx -1
c4 = n-1
c5 = c3 + anx # weird right convection boundary

# indexes for the boundaries and the corners 
#(Since arange, need to add 1 to the end point so its included. So in some way or another the end point as been increased by 1)
top_adiabatic = np.arange(1, anx-2 + 1, 1)
right_convec_fin = np.arange(((anx*2)-1),(anx*(any-1)), anx)
bottom_convec = np.arange((c_start + bnx), (c_end), 1)
right_convec_flat = np.arange((c5 + bnx), (n-bnx), bnx)
bottom_adiabatic = np.arange((n-1-(bnx-2)), (n-1), 1)
left_fixed1 = np.arange(0, (b_start+1), anx)
left_fixed2 = np.arange(b_start, (n - (bnx-1)), bnx)
left_fixed = np.concatenate ((left_fixed1, left_fixed2)) # all indexes for the left point

total_convection =  np.concatenate ((right_convec_fin, right_convec_flat, bottom_convec, [c1, c2, c3, c4, c5]))
fin_convection = np.concatenate ((right_convec_fin, bottom_convec, [c1, c2]))

# Adding the coeffs for BCs in the matrix
for i in top_adiabatic:
    A[i, i] = 2
    A[i, i+1] = -0.5
    A[i, i-1] = -0.5
    A[i, i+anx] = -1
    Su[i] = 0

for i in right_convec_fin:
    A[i, i] = Bio + 2
    A[i, i-anx] = -0.5
    A[i, i-1] = -1
    A[i, i+anx] = -0.5
    Su[i] = Bio * T_infin

for i in bottom_convec:
    A[i, i] = Bio + 2
    A[i, i-anx] = -1
    A[i, i-1] = -0.5
    A[i, i+1] = -0.5
    Su[i] = Bio * T_infin
    
for i in right_convec_flat:
    A[i, i] = Bio + 2
    A[i, i-bnx] = -0.5
    A[i, i-1] = -1
    A[i, i+bnx] = -0.5
    Su[i] = Bio * T_infin

for i in bottom_adiabatic:
    A[i, i] = 2
    A[i, i+1] = -0.5
    A[i, i-1] = -0.5
    A[i, i-bnx] = -1
    Su[i] = 0

for i in left_fixed:
    A[i,i] = 1
    Su[i] = T_sat

A[c1, c1] = Bio/2 + 1
A[c1, c1-1] = -0.5
A[c1, c1+anx] = -0.5
Su[c1] = Bio/2 * T_infin

A[c2, c2] = Bio + 1
A[c2, c2-1] = -0.5
A[c2, c2-anx] = -0.5
Su[c2] = Bio * T_infin

A[c3, c3] = Bio + 3
A[c3, c3-1] = -1
A[c3, c3+1] = -0.5
A[c3, c3-anx] = -1
A[c3, c3+anx] = -0.5
Su[c3] = Bio * T_infin

A[c4, c4] = Bio/2 + 1
A[c4, c4-1] = -0.5
A[c4, c4-bnx] = -0.5
Su[c4] = Bio * T_infin

A[c5, c5] = Bio + 2
A[c5, c5-1] = -1
A[c5, c5-anx] = -0.5
A[c5, c5+bnx] = -0.5
Su[c5] = Bio * T_infin

internal_index = np.where(np.all(A == 0, axis=1))[0] # Finds the internal nodes by finding the zero rows
internal_nodesA = internal_index[np.where(internal_index<d_start)]
internal_nodesB = internal_index[np.where(internal_index>d_end)]
internal_nodesD = np.arange(d_start, d_end+1, 1)

for i in internal_nodesA:
    A[i, i] = 4
    A[i, i-anx] = -1
    A[i, i+anx] = -1
    A[i, i-1] = -1
    A[i, i+1] = -1
    Su[i] = 0

for i in internal_nodesB:
    A[i, i] = 4
    A[i, i-bnx] = -1
    A[i, i+bnx] = -1
    A[i, i-1] = -1
    A[i, i+1] = -1
    Su[i] = 0

for i in internal_nodesD:
    A[i, i] = 4
    A[i, i-anx] = -1
    A[i, i+bnx] = -1
    A[i, i-1] = -1
    A[i, i+1] = -1
    Su[i] = 0

# ----------------------------------------Code to solve the matrices---------------------------------------#

T_1d = spsolve(A, Su)

T_2d = reshape2(T_1d, np.nan)

# Code to plot the temperature profile
X_2d = np.arange(0, anx, 1)
Y_2d = np.arange(0, ny, 1)
X_2d_plot, Y_2d_plot = np.meshgrid(X_2d, Y_2d)
plt.contourf(X_2d_plot*dx_mm, (ny-1 - Y_2d_plot)*dx_mm, T_2d, levels = 8, cmap = plt.cm.jet)
plt.title("Contour Plot of Temperature Profile")
plt.xlabel("mm")
plt.ylabel("mm")
cbar = plt.colorbar()
plt.show()

# Code to output the total heat transfer
heat_in = ( dx/dx*k* np.sum(T_1d[left_fixed] - T_1d[left_fixed+1]) )
heat_out = ( h*dx * np.sum(T_1d[total_convection] - T_infin) )
fin_heat = ( 2*h*dx * np.sum(T_1d[fin_convection] - T_infin) )
T_corner = T_1d[c3]

print(heat_in)
print(heat_out)
print(fin_heat)
print(T_corner)