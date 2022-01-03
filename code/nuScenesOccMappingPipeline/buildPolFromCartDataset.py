import numpy as np
import os
import os.path as osp
from PIL import Image
import utils.mapUtils as mapUtils
from tqdm import tqdm
import multiprocessing as mp


#===========================#  
def createImgStorDir(dirName):
    if not os.path.exists(dirName):
        try:
            os.makedirs(dirName)
        except:
            pass
    return dirName


#===========================#  
def main(logFolder):    
    # get all images inside the folder
    fileNames = [fileName for fileName in os.listdir(LOG_DIR+logFolder+"/") if fileName.endswith(".png")]
    
    # create log folder, if not already existent
    _ = createImgStorDir(LOG_DIR + logFolder +"_polar/")
    
    for fileName in tqdm(fileNames):
        # load cartesian image
        cartImg = np.asarray(Image.open(LOG_DIR + logFolder +"/"+ fileName)).astype(float)/255
        
        # image dimension in bird's eye view
        pDim = 128 # [px]
        mDim = 40. # [m]
        aRes_deg = 0.5
        aDim = int(360/aRes_deg) # [deg]
        
        # trafo image from cartesian to polar coordinates
        polImg = mapUtils.cart2polarImg(cartImg, mDim, pDim, aDim)
        
        # store polar image
        polImg = Image.fromarray((polImg*255).astype(np.uint8))
        polImg.save(LOG_DIR + logFolder +"_polar/"+ fileName[:len(logFolder)]+"_polar"+fileName[len(logFolder):])


#=========================================================================================================================================#  
LOG_DIR = "../_DATASETS_/occMapDataset/val/"

# get all folders in the log dir
namesOfLogsToConvert = ["d","ilm","ilmMapPatch","r_1","r_5","r_10","r_20","l"]
logFolders = [logFolder for logFolder in os.listdir(LOG_DIR) if (logFolder in namesOfLogsToConvert)]
logFolders.sort()

# process all log files in parallel
pool = mp.Pool(mp.cpu_count())  
pool.map(main, logFolders)   
pool.close()
pool.join()          
