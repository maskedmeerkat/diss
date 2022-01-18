import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.misc
from PIL import Image
import sys
plt.close("all")

def rayCastingImg(X):
    '''This method assumes a polar sensor model as generator for the provided
    point cloud. The function computes an image of the occupancy around the
    sensor. 
    The image is intialized with all pixel being unkown (0).    
    Afterwards, for each point in the cloud, a line from the center is computed.
    All points in between the center and the end point are marked as unoccupied
    (-1) while the end point is marked as occupied (1).
    inputs:
        X: sensorImg      
    '''  
    occImg = np.zeros(X.shape[:-1]+(1,))
    occImg[np.where(X[...,0] == 1)] = 1.0
    
    # compute the scale factors according to the angular cummulative distribution
    dPhi = 0.5 # [deg]
    varPhi = 20 # [deg]
    phiMax = 5 # [deg]
    Phi = np.arange(-phiMax-dPhi/2,phiMax+dPhi/2,dPhi) # [deg]
    cdf = norm.cdf(Phi,scale=varPhi)
    scales = cdf[1:]-cdf[:-1]
    
    Phi = np.arange(-phiMax,phiMax,dPhi)/180*np.pi # [rad]
    
    # parameters of the normal distribution around the detection
    var = .1 # [m]
    pO = 1.0
    pF = -1.0
    
    for itX, x in enumerate(X):
        # find position of center point
        centerPos = np.zeros((2,1))
        centerPos[:,0] = np.asarray(x.shape[-3:-1])/2
        
        # find position of detections
        statPos = np.asarray(np.where(x[...,0] == 1))
        dynaPos = np.asarray(np.where(x[...,1] == 1))
        
        # put all detections together and transform them into vehicle coordinates
        detectPos = np.append(statPos,dynaPos,axis=1) - centerPos - 0.5
        
        for it in range(detectPos.shape[1]):
            # compute distance from center to end point
            dist = np.linalg.norm(detectPos[:,it]) 
            
            # compute cone line for angle deviation from middle line dAngle along x axis
            line = np.zeros((2,int(np.ceil(2*dist + 4*var))))
            line[0,:] = np.arange(0,line.shape[1]/2,1/2)
            
            # compute 1D inverse sensor model logits
            logits = invSeMo_1D(line[0,:],dist,var,pO,pF)
            print(np.max(logits))

            for itPhi, phi in enumerate(Phi):
                # rotate line to end at end point
                angle_rad = np.arctan2(detectPos[1,it], detectPos[0,it]) + phi
                occImg[itX,...] = rotateLineAndAsignFreeSpace(line,scales[itPhi]*logits,centerPos,angle_rad,occImg[itX,...])
    
        print( "\r{}%".format(int( (itX+1)/X.shape[0] * 100)),end="")
        sys.stdout.flush()

    print("\n")  
             
    return occImg 

def rotateLineAndAsignFreeSpace(line,logits,centerPos,angle,occImg):
    
    R = np.matrix([ [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ])            
    line = R * line
    
    # convert line to image coordinates
    line_IC = line + centerPos
    
    # convert line to image indices
    line_IC_Idx = np.ceil(line_IC).astype(int)
    
    # filter out all indices outside the image boundaries
    correctImageIdx = np.multiply( np.multiply( (line_IC_Idx[0,:] >= 0.0), (line_IC_Idx[0,:] < occImg.shape[0]) ), 
                                   np.multiply( (line_IC_Idx[1,:] >= 0.0), (line_IC_Idx[1,:] < occImg.shape[1]) ) )
    
    correctImageIdx = np.asarray(correctImageIdx)
    line_IC_Idx = line_IC_Idx[:,correctImageIdx[0,:]]
    logits = logits[correctImageIdx[0,:]]
    
    # before setting all points along the line to unoccupied, check wether
    # occupied point is in the way
    collisionIdx = np.where(occImg[line_IC_Idx[0,:],line_IC_Idx[1,:]] <= 0.0)
    
    if (len(collisionIdx)>=2):
        if (collisionIdx[1].size):
            line_IC_Idx = line_IC_Idx[:,collisionIdx[1][:]]
            logits = logits[collisionIdx[1][:]]
        
    # update occupancy image    
    occIdx = (occImg==1)
    occImg[line_IC_Idx[0,:],line_IC_Idx[1,:],0] += logits
    occImg = np.clip(occImg,-1,1)
    occImg[occIdx] = 1
    
    return occImg
    
def invSeMo_1D(x,mean,var,pO,pF): 
    '''
        Defines an inverse sensor model with peak at (mean,pO) 
        and minimum at (mean-3*var,pF) by a 5th degree polynom 
    '''    
    # define the x coordinates of the polynom's extrema
    x1, x2, x3 = [mean-3*var,mean,mean+3*var]
    
    # define the y coordinates of the polynom's extrema and their derivatives
    Y = np.transpose(np.array([[pF,pO,0.0,0,0,0]]))
    
    # define the data matrix 
    X = np.array([[x1**5, x1**4, x1**3, x1**2, x1, 1],
                  [x2**5, x2**4, x2**3, x2**2, x2, 1],
                  [x3**5, x3**4, x3**3, x3**2, x3, 1],
                  [5*x1**4, 4*x1**3, 3*x1**2, 2*x1**1, 1, 0],
                  [5*x2**4, 4*x2**3, 3*x2**2, 2*x2**1, 1, 0],
                  [5*x3**4, 4*x3**3, 3*x3**2, 2*x3**1, 1, 0]])

    # solve the linear equation Y = X*A <=> X^-1 * Y = A
    A  = np.matmul( np.linalg.inv(X), Y)
    
    # evaluate the polynom at x
    y = A[0]*x**5 + A[1]*x**4 + A[2]*x**3 + A[3]*x**2 + A[4]*x + A[5]
    y[x<x1] = pF
    y[x>x3] = 0.0
    return y
    
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)  


print("load image")
tmp = np.array(Image.open("radar_detections.png"))[...,0]
radar_detections = np.zeros((1,128,128,2))
# radar_detections[0,:,:,0] = tmp/255
occIdx = np.where(tmp > 128)

iThOccPx = 100
radar_detections[0,occIdx[0][iThOccPx],occIdx[1][iThOccPx],0] = 1
#radar_detections = np.reshape(radar_detections,(1,)+radar_detections.shape+(1,))/255
        
print("compute inverse sensor models")
radar_ism = rayCastingImg(radar_detections)[0,:,:,0]
radar_ism = (radar_ism+1)/2 # transform range [-1,1] -> [0,1]

print("plot results")
plt.figure()
plt.subplot(1,2,1)
plt.imshow(radar_detections[0,:,:,0])
plt.subplot(1,2,2)
plt.imshow(radar_ism,cmap="gray")
plt.show()

# im = Image.fromarray((radar_ism*255).astype(np.uint8))
# im.save("radar_ism_{0:}.png".format(iThOccPx))
# im = Image.fromarray((radar_detections[0,:,:,0]*255).astype(np.uint8))
# im.save("radar_det_{0:}.png".format(iThOccPx))
    
# scipy.misc.toimage(radar_ism, cmin=0.0, cmax=1.0).save('radar_ism.png')   
    

   