
import numpy as np
import math
import matplotlib.pyplot as plt
import operator
import random
from mpl_toolkits.mplot3d import Axes3D

ml = 0
C_1_3 = .3333333333
LN_PI_2 = math.log(math.pi / 2)

def GOfX (X,M,Cov,p):
    d = X.size
    X_minus_M = X - M
    Inv_Of_Cov = np.linalg.inv(Cov)
    Det_Of_Cov = np.linalg.det(Cov)
    Dot = X_minus_M.T.dot(Inv_Of_Cov)
    G_Of_X = -0.5 * Dot.dot(X_minus_M) - (d/2.0) * LN_PI_2 - 0.5 * math.log(Det_Of_Cov)
    if not ml:
        G_Of_X = G_Of_X +np.log(p)
    return G_Of_X


def Mahalanobis_Distance(x,m,cov):
    X = np.array(x)
    M = np.array(m)
    Cov = np.array(cov)
    Inv_Of_Cov = np.linalg.inv(Cov)
    D = np.sum(np.dot(X-M,Inv_Of_Cov)*(X-M))
    return  D

def Classifier(x_vec,mu_vecs,cov_mats,Prior):
    i = 0
    g_vals = []
    for m, c in zip(mu_vecs, cov_mats):
        g_vals.append(GOfX(x_vec, m, c,Prior[i]))
        i = i + 1
    max_index, max_value = max(enumerate(g_vals), key= operator.itemgetter(1))
    return (max_value, max_index + 1)

def Predict(x,m,c,Prior):
    rows,cou  = x.shape
    output = []
    for i in range(0,cou):
        sample = np.array([[x[0][i],x[1][i],x[2][i]]]).T
        a = Classifier(sample,m,c,Prior)[1]
        output.append(a)
    return output


################ Data ################################

x1c1 = [-5.01,-5.43,1.08,0.86,-2.67,4.94,-2.51,-2.25,5.56,1.03]
x2c1 = [-8.12,-3.48,-5.52,-3.78,0.63,3.29,2.09,-2.13,2.86,-3.33]
x3c1 = [-3.68,-3.54,1.66,-4.11,7.39,2.08,-2.59,-6.94,-2.26,4.33]

x1c2 = [-0.91,1.30,-7.75,-5.47,6.14,3.60,5.37,7.18,-7.39,-7.50]
x2c2 = [-0.18,-2.06,-4.54,0.50,5.72,1.26,-4.63,1.46,1.17,-6.32]
x3c2 = [-0.05,-3.53,-0.95,3.92,-4.85,4.36,-3.65,-6.66,6.30,-0.31]

x1c3 = [5.35,5.12,-1.34,4.48,7.11,7.17,5.75,0.77,0.90,3.52]
x2c3 = [2.26,3.22,-5.31,3.42,2.39,4.33,3.97,0.27,-0.43,-0.36]
x3c3 = [8.13,-2.66,-9.87,5.19,9.21,-0.98,6.65,2.41,-8.71,6.43]

##########################################################

############ prior ###################

P_C_1 , P_C_2 , P_C_3 = .8,.1,.1

#####################################

##############3 get mean for all classes ############
M11 = np.mean(x1c1)
M12 = np.mean(x2c1)
M13 = np.mean(x3c1)

M21 = np.mean(x1c2)
M22 = np.mean(x2c2)
M23 = np.mean(x3c2)


M31 = np.mean(x1c3)
M32 = np.mean(x2c3)
M33 = np.mean(x3c3)

M1 = np.array([[M11,M12,M13]])
M2 = np.array([[M21,M22,M23]])
M3 = np.array([[M31,M32,M33]])

########################################################

##### rehape --- mean #######
z,v = M1.shape
M1 = np.reshape(M1,(v,z))
z,v = M2.shape
M2 =  np.reshape(M2,(v,z))
z,v = M3.shape
M3 = np.reshape(M3,(v,z))
##################

###### Cov ######

A = np.array([x1c1,x2c1,x3c1])
Cov1 = np.cov(A)
B = np.array([x1c2,x2c2,x3c2])
Cov2 = np.cov(B)
C = np.array([x1c3,x2c3,x3c3])
Cov3 = np.cov(C)
##################


############# Test Data ##################
Prior = [.8,.1,.1]
Mean = [M1,M2,M3]
Covariance = [Cov1,Cov2,Cov3]

X_Of_Example = [[4.54683,1,7.0343]]
X_Of_Example = np.array(X_Of_Example).T
print Classifier(X_Of_Example,Mean,Covariance,Prior)[1]

X_Of_Example = [[5,3,1]]
X_Of_Example = np.array(X_Of_Example).T
print Classifier(X_Of_Example,Mean,Covariance,Prior)[1]

X_Of_Example = [[0,0,0]]
X_Of_Example = np.array(X_Of_Example).T
print Classifier(X_Of_Example,Mean,Covariance,Prior)[1]

X_Of_Example = [[1,0,0]]
X_Of_Example = np.array(X_Of_Example).T
print Classifier(X_Of_Example,Mean,Covariance,Prior)[1]


###################################################

########## plot data ##################

fig = plt.figure()
fig.subplots_adjust(hspace=0.3)

h = .2  # step size in the mesh
# create a mesh to plot in

x_min = np.array([ min(x1c1)- 1,min(x1c2) - 1,min(x1c3) - 1]).min()
x_max = np.array([ max(x1c1)- 1,max(x1c2) - 1,max(x1c3) - 1]).max()

y_min = np.array([ min(x2c1)- 1,min(x2c2) - 1,min(x2c3) - 1]).min()
y_max = np.array([ max(x2c1)- 1,max(x2c2) - 1,max(x2c3) - 1]).max()

z_min = np.array([ min(x3c1)- 1,min(x3c2) - 1,min(x3c3) - 1]).min()
z_max = np.array([ max(x3c1)- 1,max(x3c2) - 1,max(x3c3) - 1]).max()


########################## x3 (z) fixed ######################
plt.subplot(2, 2, 1)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

zz = np.empty((xx.shape[0], xx.shape[1]))
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        zz[i,j] = 1

Z = np.zeros((xx.shape[0],xx.shape[1]))
for i in range(xx.shape[0]):
    Z[i, :] = Predict(np.c_[xx[i, :], yy[i, :], zz[i,:]].T,Mean,Covariance,Prior)

# Put the result into a color plot
levels=[0.9,1.1,1.9,2.1,2.9,3.1]
plt.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.Paired)
plt.colorbar()
plt.xlabel('X1')
plt.ylabel('X2')
##########################################################

########################## x2 (z) fixed ######################
plt.subplot(2, 2, 2)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(z_min, z_max, h))

zz = np.empty((xx.shape[0], xx.shape[1]))

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        zz[i,j] = 1

Z = np.zeros((xx.shape[0],xx.shape[1]))

for i in range(xx.shape[0]):
    Z[i, :] = Predict(np.c_[xx[i, :], zz[i,:],yy[i, :]].T,Mean,Covariance,Prior)

# Put the result into a color plot
levels=[0.9,1.1,1.9,2.1,2.9,3.1]
plt.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.Paired)
plt.colorbar()
plt.xlabel('X1')
plt.ylabel('X3')
#########################################################

######################### x1 (z) fixed ######################
plt.subplot(2, 2, 3)
xx, yy = np.meshgrid(np.arange(y_min, y_max, h),
                     np.arange(z_min, z_max, h))

zz = np.empty((xx.shape[0], xx.shape[1]))
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        zz[i,j] = 1

Z = np.zeros((xx.shape[0],xx.shape[1]))

for i in range(xx.shape[0]):
    Z[i, :] = Predict(np.c_[zz[i,:],xx[i, :],yy[i, :]].T,Mean,Covariance,Prior)

# Put the result into a color plot
levels=[0.9,1.1,1.9,2.1,2.9,3.1]
plt.contourf(xx, yy, Z, levels=levels, cmap=plt.cm.Paired)
plt.colorbar()
plt.xlabel('X3')
plt.ylabel('X2')


plt.show()


##############################################
