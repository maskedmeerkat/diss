from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/Daniel/Documents/_uni/PhD/code//_DATASETS_/occMapDataset_/val/_scenes/"
sceneName = "scene"+"0486"+"/"

yEstDirName = "shiftNet_ilmMapPatchDisc_r_20"
yEstFileName = yEstDirName + "_mapFused"

irmMap = np.array(Image.open(DATA_DIR + sceneName + "shiftNet_ilmMapPatchDisc_r_20/shiftNet_ilmMapPatchDisc_r_20_mapGeo.png"))/255
ismMap = np.array(Image.open(DATA_DIR + sceneName + yEstDirName + "/" + yEstFileName+".png"))/255

plt.figure()
N = 3
n = 1
plt.subplot(1, N, n)
n += 1
plt.imshow(irmMap)
plt.subplot(1, N, n)
n += 1
plt.imshow(ismMap)
plt.subplot(1, N, n)
n += 1
ismMap[ismMap[:, :, 2] > 0.3, :-1] = 0
ismMap[ismMap[:, :, 2] > 0.3, -1] = 1
plt.imshow(ismMap)

plt.show()
