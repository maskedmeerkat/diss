from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

DATA_DIR = "C:/Users/Daniel/Documents/_uni/PhD/diss/imgs/08_occ_mapping_exp/analysis_of_redundant_info/scene0260/ilmMap/"

ilmMap = np.array(Image.open(DATA_DIR + "ilmMap.png"))/255
occArea = ilmMap[:, :, 1]
occArea[occArea < 0.6] = 0
occArea[occArea >= 0.6] = 1

plt.figure()
N = 4
n = 1
plt.subplot(1, N, n)
n += 1
plt.imshow(ilmMap)

plt.subplot(1, N, n)
n += 1
plt.imshow(occArea)

plt.subplot(1, N, n)
n += 1
mask = (occArea*255).astype(np.uint8)
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

for c in cnts:
    cv2.drawContours(mask, [c], -1, 255, thickness=10)

plt.imshow(mask)

plt.subplot(1, N, n)
n += 1
cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

ilmMap = (ilmMap*255).astype(np.uint8)
for c in cnts:
    cv2.drawContours(ilmMap, [c], -1, (255, 255, 255), thickness=1)

plt.imshow(ilmMap)

plt.show()

cv2.imwrite(DATA_DIR + "ilmMap_contour.png", ilmMap[:, :, [2,1,0]])
