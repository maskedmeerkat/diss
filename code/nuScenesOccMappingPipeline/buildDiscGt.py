import numpy as np
import os
import os.path as osp
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp


#===========================#  
def main(fileName):  
    # load gt image
    gtImg = np.asarray(Image.open(LOG_DIR+gtFolder+fileName))
    
    # define pixel labels
    pxLabels = np.argmax(gtImg, axis=2)
    dyLabels = np.logical_and(gtImg[...,0]>120, gtImg[...,1]>120)
    
    # define discretized ground truth
    gtImg_disc = np.zeros_like(gtImg, dtype=np.uint8)
    gtImg_disc[pxLabels==0,0] = 255
    gtImg_disc[pxLabels==1,1] = 255
    gtImg_disc[pxLabels==2,2] = 255
    gtImg_disc[dyLabels,0] = 127
    gtImg_disc[dyLabels,1] = 127
    gtImg_disc[dyLabels,2] = 0
    
    # store discretized gt image
    gtImg_disc = Image.fromarray(gtImg_disc)
    gtImg_disc.save(LOG_DIR+discGtFolder+"ilmMapPatchDisc"+fileName[len("ilmMapPatch"):])


#=========================================================================================================================================#  
LOG_DIR = "../_DATASETS_/occMapDataset/val/"
gtFolder = "ilmMapPatch/"
discGtFolder = "ilmMapPatchDisc/"
os.makedirs(LOG_DIR+discGtFolder,exist_ok = True)

# get all folders in the log dir
fileNames = [fileName for fileName in os.listdir(LOG_DIR+gtFolder) if fileName.endswith(".png")]

# process all log files in parallel
pool = mp.Pool(mp.cpu_count())  
pool.map(main, fileNames)   
pool.close()
pool.join()          
