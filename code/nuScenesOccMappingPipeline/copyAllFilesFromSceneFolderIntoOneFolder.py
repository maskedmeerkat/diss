# import libraries
from distutils.dir_util import copy_tree
import os
from tqdm import tqdm

#============================MAIN============================================#
# load log directories
LOG_DIR = "../_DATASETS_/occMapDataset/val/"
LOG_DIR_OLD = LOG_DIR + "_scenes/"
dirNames = [dirName for dirName in os.listdir(LOG_DIR_OLD) if dirName.startswith('scene')]
dirNames.sort()

# go thru all directories
for iDir in tqdm(range(len(dirNames))):
    # copy next dir
    dirName = dirNames[iDir]
    copy_tree(LOG_DIR_OLD + dirName, LOG_DIR)
    
    
