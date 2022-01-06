from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

DATA_DIR = "C:/Users/Daniel/Documents/_uni/PhD/code//_DATASETS_/occMapDataset/val/_scenes/"
sceneName = "scene"+"0486"+"/"

ilmMap = np.array(Image.open(DATA_DIR + sceneName + "ilmMap/ilmMap.png"))/255
occArea = ilmMap[:, :, 1]
occArea[occArea < 0.6] = 0
occArea[occArea >= 0.6] = 1

plt.figure()
N = 3
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
    cv2.drawContours(mask, [c], -1, 255, thickness=15)
plt.imshow(mask)

plt.show()
