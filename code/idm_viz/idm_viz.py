import numpy as np
import tikzplotlib
from scipy.special import erf
import matplotlib.pyplot as plt
plt.close("all")
SQRT_2 = np.sqrt(2)


def gauss(x, mu, sig):
    return 1/(np.sqrt(2*np.pi)*sig) * np.exp(- (x-mu)**2/(2*sig**2))


def q_bspline(x):
    '''
    Define the cdf of a quadratic b-spline that approximates the Gaussian
    distribution.
    '''
    if not isinstance(x, type(np.zeros(3))):
        x = np.array([x])
    y = np.zeros(x.shape)
    cond = x < -3
    y[cond] = 0

    cond = np.logical_and(x >= -3, x <= -1)
    y[cond] = 1 / 48 * (3 + x[cond]) ** 3

    cond = np.logical_and(x > -1, x < 1)
    y[cond] = 1 / 2 + 1 / 24 * x[cond] * (3 + x[cond]) * (3 - x[cond])

    cond = np.logical_and(x >= 1, x <= 3)
    y[cond] = 1 - 1 / 48 * (3 - x[cond]) ** 3

    cond = x > 3
    y[cond] = 1

    return y


def idm_bspline(x):
    return q_bspline(x) - 0.5 * q_bspline(x - 3)


def idm_ideal(r, R, tau):
    if tau == 0:
        p = np.zeros_like(r)
        dr = np.abs(r-R)
        p[np.argmin(dr)] = 1
        p[np.argmin(dr)+1:] = .5
    else:
        p = np.ones_like(r)
        p[r < R] = 0
        p[r > R+tau] = 0.5
    return p


def idm_gauss(r, R, sig_r, tau):
    # dr = r[1] - r[0]
    # p_ideal = idm_ideal(r, R, 0)
    # return convolve(p_ideal, gauss(np.linspace(-8*sig_r, 8*sig_r, 1000), 0, sig_r))
    return 0.5 * erf((r-R)/(SQRT_2*sig_r)) - 0.25 * erf((r-R-tau)/(SQRT_2*sig_r)) + 0.25


r = np.arange(0, 5, 0.01)
R = 2
sigma = 0.1
tau = 3*sigma

p_ideal = idm_ideal(r, R, tau)


plt.figure(figsize=(7, 3))
plt.subplot(121)
plt.step(r, p_ideal, "k-", where="post")
plt.plot(r, idm_gauss(r, R, sigma, tau)[:r.shape[0]], "g")
plt.plot(r, idm_bspline((r - R)/sigma), "b--")
plt.plot(r, 0.5*np.ones_like(r), "r")
plt.legend(["IDM$_{ideal}$", "IDM$_{Gauss}$", "IDM$_{B-Spline}$", "fr-oc-border"], loc="lower right")
plt.yticks(rotation=90)
plt.xlabel("r")
plt.ylabel("p$_o$")

plt.subplot(122)
plt.step(r, p_ideal, "k-", where="post")
plt.plot(r, idm_gauss(r, R, sigma, tau)[:r.shape[0]], "g")
plt.plot(r, idm_bspline((r - R) / sigma), "b--")
plt.plot(r, 0.5 * np.ones_like(r), "r")
plt.legend(["IDM$_{ideal}$", "IDM$_{Gauss}$", "IDM$_{B-Spline}$", "fr-oc-border"])
dx = 0.001
plt.xlim([R-dx, R+dx])
dy = 0.001
plt.ylim([0.5-dy, 0.5+dy])
plt.xticks([1.999, 2.000, 2.001], labels=["1.999", "2.000", "2.001"])
plt.yticks([0.501, 0.500, 0.499], rotation=45)
plt.xlabel("r")

plt.margins(0, 0)
plt.savefig("../../imgs/05_state_of_the_art/geo_isms/radial_geo_idm.pgf", bbox_inches='tight', pad_inches=0)

plt.show()



