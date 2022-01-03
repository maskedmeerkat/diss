print("# load libraries")
from PIL import Image
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
from tensorflow.python.platform import gfile

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

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


# DirNet
# deepIsmModel = "dirNet"
# deepIsmModel = "shiftNet"
# deepIsmModel = "softNet"
deepIsmModel = "occNet"
# MODEL_NAMES = [modelName for modelName in os.listdir("./models/exp_deep_ism_comparison/") if (modelName.endswith(".pb") and modelName.startswith(deepIsmModel))]
# MODEL_NAMES = [modelName for modelName in os.listdir("./models/exp_network_tuning/") if (modelName.endswith(".pb") and ("r_1" in modelName or "r_20" in modelName) and modelName.startswith(deepIsmModel))]
MODEL_NAMES = ["occNet_100_xs_ilmMapPatch_r_1__20201211_152215_ckpt_54.pb",
               "occNet_125_ilmMapPatch_r_1__20201206_113913_ckpt_538.pb"]

LOG_DIR = "../_DATASETS_/occMapDataset/val/_scenes/"

possibleInputNames = ["r_1","r_5","r_10","r_20","l","d","c","lr20","dr20","s","irm_1","irm_20"]

for MODEL_NAME in MODEL_NAMES:
    print("\n")
    print(MODEL_NAME)
    
    # clear old graphs that might be still stored in e.g. ipython console
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    # plt.close("all")
    
    MODEL_DIR = "./models/exp_network_tuning/" + MODEL_NAME
    xClassName = ""
    for possibleInputName in possibleInputNames:
        if "_" + possibleInputName + "__" in MODEL_NAME:
            xClassName = possibleInputName
    with tf.Session() as sess, gfile.FastGFile(MODEL_DIR,'rb') as graphFile:
        # restore the graph from the pb file
        x, y_fake = restoreGraphFromPB(sess, graphFile)
    
        sceneNames = [sceneName for sceneName in os.listdir(LOG_DIR) if sceneName.startswith("scene")]
        for sceneName in tqdm(sceneNames):
            DATA_PATH = LOG_DIR + sceneName + "/"
                    
            RESULT_PATH = DATA_PATH + MODEL_DIR.split("/")[-1].split("__")[0] + "/"
            os.makedirs(RESULT_PATH, exist_ok=True) 
            grayScaleInputs = ["r_1", "r_5", "r_10", "r_20", "l", "d"]
            
            # load all file names
            fileNames = [fileName for fileName in os.listdir(DATA_PATH+xClassName+"/") if (fileName.endswith(".png") and fileName.startswith(xClassName))]
            
            # perform inference for each file 
            for fileName in fileNames:
                # load image
                x_VAL_IMG_PATH = DATA_PATH+xClassName+"/"+fileName
                if xClassName in grayScaleInputs:
                    inImg = np.zeros((1,128,128,1))
                    inImg[0,:,:,0] = np.array(Image.open(x_VAL_IMG_PATH))/255
                else:
                    inImg = np.zeros((1,128,128,3))
                    inImg[0,:,:,:] = np.array(Image.open(x_VAL_IMG_PATH))/255
                
                # perform inference                
                outImg = sess.run(y_fake, feed_dict = {x: inImg})[0,...]
            
                # plt.figure()
                # plt.subplot(1,4,1)
                # plt.imshow(inImg[0,:,:,0],cmap="gray")
                # plt.subplot(1,4,2)
                # plt.imshow(outImg[0,:,:,0],cmap="gray")
                # plt.subplot(1,4,3)
                # plt.imshow(outImg[0,:,:,1],cmap="gray")
                # plt.subplot(1,4,4)
                # plt.imshow(outImg[0,:,:,2],cmap="gray")               
                # outImg = np.append(outImg[...,0], np.append(outImg[...,1],outImg[...,2],axis=1),axis=1)
                y = Image.fromarray((outImg*255).astype(np.uint8))
                y.save(RESULT_PATH + deepIsmModel + "_" + fileName)