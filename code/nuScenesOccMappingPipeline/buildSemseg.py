import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# clear old graphs that might be still stored in e.g. ipython console
#tf.reset_default_graph()
import numpy  as np
from PIL import Image
from tqdm import tqdm
#from nuscenes.nuscenes import NuScenes
import os
import time


LOG_DIR = "./models/deeplab_cityscapes_xception71_trainfine_2018_09_08/frozen_inference_graph.pb" 

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

print('\n#====================================================================#')
print('# Restore frozen Graph')
print('#====================================================================#')
graph = load_graph(LOG_DIR)

print("# get inpute name")
numOps = len(graph.get_operations())
for it, op in enumerate(graph.get_operations()):
    if (it==0):
        inName = op.name+":0"
        print("    Input Name:",inName)
    elif (it==(numOps-1)):
        outName = op.name+":0"
        print("    Output Name:",outName)
    
print("# access the input and output nodes")
x = graph.get_tensor_by_name(inName)
y = graph.get_tensor_by_name(outName)  

# create nuscenes object      
#DATA_DIR = '../_DATASETS_/NuScenes/samples/'
DATA_DIR = '../_DATASETS_/NuScenes/sweeps/'
#nusc = NuScenes(version='v1.0-mini', dataroot=DATA_DIR, verbose=False)      

# get number of front cam images
camNames = ["CAM_BACK/", "CAM_BACK_LEFT/", "CAM_BACK_RIGHT/",
            "CAM_FRONT/", "CAM_FRONT_LEFT/", "CAM_FRONT_RIGHT/",]

print("# start tensorflow session")      
with tf.Session(graph=graph) as sess:
    for camName in camNames:
        print("\n#====================================#")
        print("# Perform semSeg for "+camName)
        print("#====================================#")
              
        # create directory to store semantic segmentation results to
        semSegStorDir = DATA_DIR + "SEMSEG_" + camName
        if not os.path.exists(semSegStorDir):
            os.makedirs(semSegStorDir)
              
        # load all file names for current camera
        fileNames = [f for f in os.listdir(DATA_DIR + camName) if f.endswith('.jpg')]
        
        for fileName in tqdm(fileNames):
            t0 = time.time()
            
            # load input image from front cam
            img = np.asarray( Image.open(DATA_DIR + camName + fileName) )
            
            # perform inference
            feed_dict={x: [img]}
            segResult = sess.run(y,feed_dict)[0,...]
            
            # store the segmentation result
            segResult = Image.fromarray( segResult.astype(np.uint8) )
            segResult.save(semSegStorDir + fileName[:31] + "SEMSEG_" + fileName[31:] )
            
            # print progress
            dt = time.time()-t0
        


