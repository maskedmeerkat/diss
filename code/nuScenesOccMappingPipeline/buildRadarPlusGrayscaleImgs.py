import numpy as np
import os
import os.path as osp
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp


# #===========================#  
# def main(timestamp):  
#     # load radar & grayscale image
#     rImg_ = np.asarray(Image.open(LOG_DIR+rFolder +rFolder[:-1] +timestamp))
#     grImg = np.asarray(Image.open(LOG_DIR+grFolder+grFolder[:-1]+timestamp))[...,np.newaxis]
    
#     # move dynamic and static pixels into separate channels
#     rImg = np.zeros(rImg_.shape+(2,),dtype=np.uint8)
#     rImg[rImg_>250,0] = 255
#     rImg[(rImg_>0) * (rImg_<250),1] = 255
    
#     # combine radar with other grayscale image
#     rGrImg = np.append(grImg, rImg, axis=2)
    
#     # store discretized gt image
#     rGrImg = Image.fromarray(rGrImg)
#     rGrImg.save(LOG_DIR+combFolder+combFolder[:-1]+timestamp)


# #=========================================================================================================================================#  
# LOG_DIR = "../_DATASETS_/occMapDataset/train/"
# # radar folder name
# rFolder = "r_20/"
# # grayscale folder name
# grFolder = "l/"
# combFolder = "lr20/"
# os.makedirs(LOG_DIR+combFolder,exist_ok = True)

# # get all files in the log dir
# timestamps  = [fileName[len(rFolder[:-1]):] for fileName in os.listdir(LOG_DIR+rFolder ) if fileName.endswith(".png")]

# # process all log files in parallel
# pool = mp.Pool(mp.cpu_count())  
# pool.map(main, timestamps)   
# pool.close()
# pool.join()    


#===========================#  
def main(sceneName):  
    # create combined image folder
    os.makedirs(LOG_DIR+sceneName+combFolder,exist_ok = True)

    # loop thru all samples    
    timestamps  = [fileName[len(rFolder[:-1]):] for fileName in os.listdir(LOG_DIR+sceneName+rFolder) if fileName.endswith(".png")]    
    for timestamp in timestamps:
        # load radar & grayscale image
        rImg_ = np.asarray(Image.open(LOG_DIR+sceneName+rFolder +rFolder[:-1] +timestamp))
        grImg = np.asarray(Image.open(LOG_DIR+sceneName+grFolder+grFolder[:-1]+timestamp))[...,np.newaxis]
        
        # move dynamic and static pixels into separate channels
        rImg = np.zeros(rImg_.shape+(2,),dtype=np.uint8)
        rImg[rImg_>250,0] = 255
        rImg[(rImg_>0) * (rImg_<250),1] = 255
        
        # combine radar with other grayscale image
        rGrImg = np.append(grImg, rImg, axis=2)
        
        # store discretized gt image
        rGrImg = Image.fromarray(rGrImg)
        rGrImg.save(LOG_DIR+sceneName+combFolder+combFolder[:-1]+timestamp)


#=========================================================================================================================================#  
LOG_DIR = "../_DATASETS_/occMapDataset/val/_scenes/"
# radar folder name
rFolder = "r_20/"
# grayscale folder name
grFolder = "d/"
combFolder = "dr20/"

# get all files in the log dir
sceneNames  = [sceneName+"/" for sceneName in os.listdir(LOG_DIR) if sceneName.startswith("scene")]

# process all log files in parallel
pool = mp.Pool(mp.cpu_count())  
pool.map(main, sceneNames)   
pool.close()
pool.join()       
