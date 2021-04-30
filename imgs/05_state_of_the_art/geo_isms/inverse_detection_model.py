import numpy as np
import matplotlib.pyplot as plt
import math 
plt.close("all")

def Q_cdf(x):
    '''
    Define the cdf of a quadratic b-spline that approximates the Gaussian 
    distribution.
    '''
    y = np.zeros(x.shape)
    cond = x < -3
    y[cond] = 0
    
    cond = np.logical_and(x >= -3, x <= -1)
    y[cond] = 1/48*(3+x[cond])**3
    
    cond = np.logical_and(x > -1, x < 1)
    y[cond] = 1/2 + 1/24*x[cond]*(3+x[cond])*(3-x[cond])
    
    cond = np.logical_and(x >= 1, x <= 3)
    y[cond] = 1 - 1/48*(3-x[cond])**3
    
    cond = x > 3
    y[cond] = 1
    
    return y


def H(x):
    return Q_cdf(x) - 0.5*Q_cdf(x-3)

sig_r = 0.1
r_ = 1.
tau = 1*sig_r
r = np.linspace(0,r_+10*sig_r,1000)

p1 = np.zeros_like(r)
for i in range(r.shape[0]):
    p1[i] = 0.5*math.erf((r_ + r[i]-2)/np.sqrt(2)/sig_r) - 1/4*math.erf((r_ + r[i] - tau -2)/np.sqrt(2)/sig_r) + 1/4

p2 = H( (r-r_)/sig_r )

plt.figure()
plt.plot(r, p1)
plt.plot(r, p2)
plt.plot([r_,r_],[0,0.5],"k--")
plt.plot([0,r_],[0.5,0.5],"k--")
plt.xlim([0,np.max(r)])
plt.ylim([0,1])