print("# Import Libraries")
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
import time
from skimage.metrics import structural_similarity as sk_ssim
from utils.prettytable import PrettyTable
import csv
import numpy as np
from tqdm import tqdm
from PIL import Image
import tensorflow.compat.v1 as tf
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile
import cv2


# clear old graphs that might be still stored in e.g. ipython console
tf.reset_default_graph()
tf.keras.backend.clear_session()



def restoreGraphFromPB(sess, graphFile):
    '''
    Restore the "interface" variables of the graph to perform inference.
    '''
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graphFile.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def,
                        input_map=None,
                        return_elements=None,
                        name="",
                        op_dict=None,
                        producer_op_list=None)
    return getGraphVariables()


def getGraphVariables(): 
    '''
    Get the input and output nodes of the model.
    '''
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("input_1:0")    
    y_fake = graph.get_tensor_by_name("output_0:0")
    
    return x, y_fake


# ============================================================================= #
print("\n# Define Parameters")
# data directory
DATA_DIR = "../_DATASETS_/occMapDataset_/train/_scenes/"

# for network verification
possibleInputNames = ["r_1","r_5","r_10","r_20","l","d","c","lr20","dr20","s","irm_1","irm_20"]
MODEL_DIR = "./models/exp_network_tuning/"
MODEL_NAMES = [MODEL_NAME for MODEL_NAME in os.listdir(MODEL_DIR) if MODEL_NAME.endswith(".pb")]
# MODEL_NAMES = ["dirNet_ilmMapPatchDisc_r_1__20201223_215714_ckpt_180.pb",
#                "shiftNet_ilmMapPatchDisc_r_1__20201223_215050_ckpt_198.pb",
#                "softNet_ilmMapPatchDisc_r_1__20201223_215415_ckpt_688.pb"]
# MODEL_NAMES = ["occNet_075_ilmMapPatch_r_1__20201203_095051_ckpt_73.pb",
#                "occNet_100_ilmMapPatch_r_1__20201203_094530_ckpt_348.pb"]
# MODEL_NAMES = ["_irm_1__", "_irm_20__"]
# MODEL_NAMES.sort()
# MODEL_NAMES = MODEL_NAMES[7:]

# MODEL_NAMES = ["irm_1"]

for MODEL_NAME in MODEL_NAMES:
    # retrieve input name from ckpt name
    inputName = ""
    for possibleInputName in possibleInputNames:
        if "_"+possibleInputName+"__" in MODEL_NAME:
            inputName = possibleInputName
    if inputName == "":
        inputName = MODEL_NAME

    print("\n# Compute Metrics for ", MODEL_NAME)
    # get all directory names of scene data
    sceneNames = [sceneName for sceneName in os.listdir(DATA_DIR) if sceneName.startswith('scene')]
    sceneNames.sort()

    # store mean inference time on gpu and cpu
    mInfTime_gpu = -1
    mInfTime_cpu = -1

    # clear old graphs that might be still stored in e.g. ipython console
    tf.reset_default_graph()
    tf.keras.backend.clear_session()

    with tf.Session() as sess:
        if MODEL_NAME[-3:] == ".pb":
            graphFile = gfile.FastGFile(MODEL_DIR + MODEL_NAME, 'rb')
            x, y_fake = restoreGraphFromPB(sess, graphFile)

        for iScene in tqdm(range(int(len(sceneNames)))):
        # for iScene in [0]:
            # get current scene
            sceneName = sceneNames[iScene]

            # loop thru all imgs in the scene and compute the metric
            xFiles = os.listdir(DATA_DIR + sceneName + "/" + inputName + "/")
            xFiles = [xFile for xFile in xFiles if xFile.startswith(inputName+'__')]
            xFiles.sort()

            # create folder to store results to
            curr_stor_path = os.path.join(DATA_DIR, sceneName, MODEL_NAME.split("__")[0])
            os.makedirs(curr_stor_path, exist_ok=True)

            for iSample in range(len(xFiles)):
                x_ = np.array(Image.open(DATA_DIR + sceneName + "/"+inputName+"/" + xFiles[iSample]))

                # trafo inputs from [0,255] -> [0,1]
                if len(x_.shape) == 2:
                    x_ = x_[np.newaxis, :, :, np.newaxis]/255
                else:
                    x_ = x_[np.newaxis, :, :, :]/255

                # perform inference
                y_est = sess.run(y_fake, feed_dict={x: x_})[0, ...]

                # store the result
                y_est = Image.fromarray((y_est*255).astype(np.uint8))
                y_est.save(os.path.join(curr_stor_path, "{0:05}.png".format(iSample)))
                

root_folder_path = os.path.join(DATA_DIR, "scene0042")
folder_names = [file for file in os.listdir(root_folder_path) if os.path.isdir(os.path.join(root_folder_path, file))]
img_separator = np.ones((128, 10, 3))*255

for iImage in range(len(os.listdir(os.path.join(root_folder_path, folder_names[0])))):
    img_stiched = img_separator
    # load and stich images of same index from all folders
    for iPlt, folder_name in enumerate(folder_names):
        image_names = os.listdir(os.path.join(root_folder_path, folder_name))
        image_path = os.path.join(root_folder_path, folder_name, image_names[iImage])
        img = np.asarray(Image.open(image_path))
        if len(img.shape) == 2:
            img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
        img_stiched = np.append(img_stiched, img, axis=1)
        img_stiched = np.append(img_stiched, img_separator, axis=1)
    # store stiched image
    img_stiched = Image.fromarray(img_stiched.astype(np.uint8))
    img_stiched.save(os.path.join(root_folder_path, "{0:05}.png".format(iImage)))

    
        
        
        
        
        
        
        
        
        
        
    