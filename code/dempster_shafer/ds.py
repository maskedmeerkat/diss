import numpy as np
import matplotlib.pyplot as plt
import sys
#import matplotlib as mpl
#mpl.use('pdf')
plt.close('all')


def plotDsOverTime(m, title):
    time = range(m.shape[1])
    plt.plot(time,m[0,:],"g")
    plt.plot(time,m[1,:],"r")
    plt.plot(time,m[2,:],"b")
    
    plt.ylim([-0.1,1.1])
    plt.xlim([0., m.shape[1]])
    plt.ylabel(title)
    plt.legend(["free","occupied","unknown"])


def limitCertainty(m,u_min):
    # unknown mass cannot fall beneath u_min
    m[:-1] = (1-u_min)*m[:-1]
    m[2] = 1 - m[0] - m[1]
    return m


def fuseMasses(m1, m2, combRule = 0, uMin=0.0, entropyScaling=False, eps=1e-8,convergFactor=1.):
    '''
    Fuses two masses defined as m = [{fr},{oc},{fr,oc}] according to the
    Dempster's Rule of Combination.
    
    combRule: 
        0 = Yager
        1 = Yager with konflict being assigned to free and occ equally
        2 = Dempster
    '''
    if (entropyScaling):
        # FIRST: rescale mass to have at least uMin unknown mass        
        m = limitCertainty(m1,uMin)
        
        # SECOND: rescale mass to account for redundant information        
        # current conflict 
        k = m1[0]*m2[1] + m1[1]*m2[0]
        # difference in information 
        h1_2 = np.clip(convergFactor*(m2[2] - m1[2]), 0., 1.)        
        
        # limit the difference in information
        h1_2_max = np.clip((uMin - m2[2]) / (m2[2]*m1[2] - m2[2] + k), 0., 1.)
        
        if (m2[2]*m1[2] - m2[2] + k) < 0:
            h1_2 = np.min((h1_2, h1_2_max))
        else:
            h1_2 = np.max((h1_2, h1_2_max))
        if h1_2 < 0.001:
            h1_2=0
        
        # scale fr and occ masses
        m1[:-1] *= h1_2
        m1[2] = 1. - np.sum(m1[:-1])
    
    # limit certainty (needed for Dempster comb rule)
    m2 = limitCertainty(m2, eps)
    
    # compute the conflict
    k = m1[0]*m2[1] + m1[1]*m2[0]
    
    # compute the fused masses of each class
    m = np.zeros((3,1))
    
    # Yager
    if (combRule == 0):
        m[0] = (m1[0]*m2[0] + m1[0]*m2[2] + m1[2]*m2[0])
        m[1] = (m1[1]*m2[1] + m1[1]*m2[2] + m1[2]*m2[1])
        m[2] = m1[2]*m2[2] + k
    # modified Yager (conflict equally assigned to fr & occ classes)
    elif (combRule == 1):
        m[0] = m1[0]*m2[0] + m1[0]*m2[2] + m1[2]*m2[0] + k/2
        m[1] = m1[1]*m2[1] + m1[1]*m2[2] + m1[2]*m2[1] + k/2
        m[2] = m1[2]*m2[2]
    # Dempster
    else:                
        m[0] = (m1[0]*m2[0] + m1[0]*m2[2] + m1[2]*m2[0])/(1 - k)
        m[1] = (m1[1]*m2[1] + m1[1]*m2[2] + m1[2]*m2[1])/(1 - k)
        m[2] = (m1[2]*m2[2])/(1 - k)
    
    # normalize
    m[0] = np.clip(m[0], 0., 1.)
    m[1] = np.clip(m[1], 0., 1.)
    m[2] = 1 - m[0] - m[1]
    
    return m
#============================MAIN=============================================#
# parameters
uMin = 0.2
ampFactor = 10.0

# define masses as m = [{fr},{oc},{fr,oc}] 
numMeas = 500
numMeasPart = int(numMeas//10)
mMeas = np.zeros((3,numMeas))
mMeas[0,0*numMeasPart:1*numMeasPart] = 0.0
mMeas[1,0*numMeasPart:1*numMeasPart] = 0.4
mMeas[2,0*numMeasPart:1*numMeasPart] = 0.6

mMeas[0,1*numMeasPart:2*numMeasPart] = 0.0
mMeas[1,1*numMeasPart:2*numMeasPart] = 0.7
mMeas[2,1*numMeasPart:2*numMeasPart] = 0.3

mMeas[0,2*numMeasPart:3*numMeasPart] = 1.0
mMeas[1,2*numMeasPart:3*numMeasPart] = 0.0
mMeas[2,2*numMeasPart:3*numMeasPart] = 0.0

mMeas[0,3*numMeasPart:4*numMeasPart] = 0.1
mMeas[1,3*numMeasPart:4*numMeasPart] = 0.5
mMeas[2,3*numMeasPart:4*numMeasPart] = 0.4

mMeas[0,4*numMeasPart:5*numMeasPart] = 0.9
mMeas[1,4*numMeasPart:5*numMeasPart] = 0.0
mMeas[2,4*numMeasPart:5*numMeasPart] = 0.1

mMeas[0,5*numMeasPart:6*numMeasPart] = 0.0
mMeas[1,5*numMeasPart:6*numMeasPart] = 0.9
mMeas[2,5*numMeasPart:6*numMeasPart] = 0.1

mMeas[0,6*numMeasPart:] = 0.5
mMeas[1,6*numMeasPart:] = 0.5
mMeas[2,6*numMeasPart:] = 0.0

#mMeas[0,6*numMeasPart:7*numMeasPart] = 0.5
#mMeas[1,6*numMeasPart:7*numMeasPart] = 0.5
#mMeas[2,6*numMeasPart:7*numMeasPart] = 0.0
#
#mMeas[0,7*numMeasPart:8*numMeasPart] = 0.9
#mMeas[1,7*numMeasPart:8*numMeasPart] = 0.0
#mMeas[2,7*numMeasPart:8*numMeasPart] = 0.1
#
#mMeas[0,8*numMeasPart:9*numMeasPart] = 0.0
#mMeas[1,8*numMeasPart:9*numMeasPart] = 0.9
#mMeas[2,8*numMeasPart:9*numMeasPart] = 0.1
#
#mMeas[0,9*numMeasPart:10*numMeasPart] = 0.9
#mMeas[1,9*numMeasPart:10*numMeasPart] = 0.0
#mMeas[2,9*numMeasPart:10*numMeasPart] = 0.1

mUnDiff = np.zeros((3,numMeas))
mYager_ = np.zeros((3,numMeas))
mYager = np.zeros((3,numMeas))
mDempster = np.zeros((3,numMeas))
p1 = np.zeros((numMeas,))
p12 = np.zeros((numMeas,))
h1 = np.zeros((numMeas,))
h12 = np.zeros((numMeas,))
m0 = np.array([[0.], [0.], [1.]])

for it in range(numMeas):        
    # fuse the fused signal with the last prediction
    if (it==0):
        mUnDiff[:,[it]] = m0
        mYager_[:,[it]] = m0
        mYager[:,[it]] = m0
        mDempster[:,[it]] = m0
    else:
        mUnDiff[:,[it]] = fuseMasses(mMeas[:,[it]].copy(), mUnDiff[:,it-1] ,uMin=uMin, convergFactor=ampFactor,entropyScaling=True, combRule=0)
        mYager_[:,[it]] = fuseMasses(mMeas[:,[it]].copy(), mYager_[:,it-1], combRule=1)
        mYager[:,[it]] = fuseMasses(mMeas[:,[it]].copy(), mYager[:,it-1], combRule=0)
        mDempster[:,[it]] = fuseMasses(mMeas[:,[it]].copy(), mDempster[:,it-1], combRule=2)
        
print("")  
    
#plt.rc('font', family='serif', serif='Times')
#plt.rc('text', usetex=True)
#plt.rc('xtick', labelsize=8)
#plt.rc('ytick', labelsize=8)
#plt.rc('axes', labelsize=8)

numSubPlts = 5
subPltIdx = 0
fig = plt.figure()
subPltIdx += 1
plt.subplot(numSubPlts,1,subPltIdx)
plotDsOverTime(mMeas, "Input Masses")

subPltIdx += 1
plt.subplot(numSubPlts,1,subPltIdx)
plotDsOverTime(mUnDiff, "Unknown Diff")
plt.plot([0, numMeas],[uMin, uMin],"b--")
plt.legend(["free","occupied","unknown","uMin"])

subPltIdx += 1
plt.subplot(numSubPlts,1,subPltIdx)
plotDsOverTime(mYager_, "with modified Yager")
plt.plot([0, numMeas],[uMin, uMin],"b--")
plt.legend(["free","occupied","unknown","uMin"])

subPltIdx += 1
plt.subplot(numSubPlts,1,subPltIdx)
plotDsOverTime(mYager, "with Yager")
plt.plot([0, numMeas],[uMin, uMin],"b--")
plt.legend(["free","occupied","unknown","uMin"])

subPltIdx += 1
plt.subplot(numSubPlts,1,subPltIdx)
plotDsOverTime(mDempster, "with Dempster")
plt.plot([0, numMeas],[uMin, uMin],"b--")
plt.legend(["free","occupied","unknown","uMin"])




