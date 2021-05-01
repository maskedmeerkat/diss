import numpy as np

m1 = np.array([0.0, 0.0, 1.0])
m2 = np.array([0.0, 1.0, 0.0])
m3 = np.array([0.5, 0.5, 0.0])
m4 = np.array([0.1, 0.7, 0.2])

p_x  = 0.5
p_xy = 1. - 1e-6

H_x  = - p_x  * np.log2(p_x) - (1-p_x) * np.log2(1-p_x)
H_xy  = - p_xy  * np.log2(p_xy) - (1-p_xy) * np.log2(1-p_xy)

H_y_x = H_xy - H_x