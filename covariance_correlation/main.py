import numpy as np
n = 100
x = np.random.normal(size=n)

#print(np.mean(x))
y = 4 + 3*x+np.random.normal(size=n)

W = np.vstack([x,y])
C1 = np.cov(W)
C2 = np.cov([x,y])
#It seems that instead of using matrix W you could also just write C = np.cov([x,y])

#print(C1)
#print(C2)


x = x - np.mean(x)
y = y - np.mean(x)

variance_x = np.sum(x@x)/n
variance_y = np.sum(y@y)/n

cov_xy = np.sum(x@y)/n
cov_xx = np.sum(x@x)/n
cov_yy = np.sum(y@y)/n

C = np.zeros([2,2])
C[0,0]= cov_xx/variance_x
C[1,1]= cov_yy/variance_y
C[0,1]= cov_xy/np.sqrt(variance_y*variance_x)
C[1,0]= C[0,1]
#print(C)

#We could also do this in pandas!
import pandas as pd
X = (np.vstack([x,y])).T

Xpd = pd.DataFrame(X)
correlation_matrix = Xpd.corr()

print(correlation_matrix) #Damn it makes so much eaaasier with pandas!
