from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "C:/Users/Daniel/Documents/_uni/PhD/diss/imgs/08_occ_mapping_exp/analysis_of_prior/scene0488/"

yEstDirName = "shiftNet_ilmMapPatchDisc_r_20"
yEstFileName = yEstDirName + "_mapFused_prior"

irmMap = np.array(Image.open(DATA_DIR + "irmMap/irmMap.png"))/255
ismMap = np.array(Image.open(DATA_DIR + yEstDirName + "/" + yEstFileName+".png"))/255

# only keep soley initialized area
ismMap_init = ismMap.copy()
ismMap_init[ismMap_init[:, :, -1] <= 0.29, :-1] = 0
ismMap_init[ismMap_init[:, :, -1] <= 0.29, -1] = 1

# plt.figure()
# N = 3
# n = 1
# plt.subplot(1, N, n)
# n += 1
# plt.imshow(irmMap)
# plt.subplot(1, N, n)
# n += 1
# plt.imshow(ismMap)
# plt.subplot(1, N, n)
# n += 1
# plt.imshow(ismMap)
#
# plt.show()

Image.fromarray((ismMap_init*255).astype(np.uint8)).save(DATA_DIR + yEstDirName + "/" + yEstFileName+"__init.png")
