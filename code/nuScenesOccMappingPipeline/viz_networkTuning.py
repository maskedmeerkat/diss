import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mtick
import numpy as np
import csv
import os
plt.close("all")

MODEL_DIR = "./models/exp_network_tuning/"
logFileNames = [x[0] for x in os.walk(MODEL_DIR)][1:]
logFileNames.sort()

gpuTimes = [0]*len(logFileNames)
cpuTimes = [0]*len(logFileNames)
modelSizes = np.asarray([0]*len(logFileNames)).astype(float)
mIoUs = [0]*len(logFileNames)
best_mIoUs = np.asarray([0]*len(logFileNames)).astype(float)
mCpuTimes = np.asarray([0]*len(logFileNames)).astype(float)
for iFile, logFileName in enumerate(logFileNames):
    csvFiles = [csvFile for csvFile in os.listdir(logFileName) if csvFile.endswith(".csv")]
    
    # load stats for each model
    gpuTimes[iFile] = np.zeros(len(csvFiles))
    cpuTimes[iFile] = np.zeros(len(csvFiles))    
    mIoUs[iFile] = np.zeros(len(csvFiles))
    for iCsvFile in range(len(csvFiles)):
        stats = []
        # print(os.path.exists(logFileName+"/"+csvFiles[iCsvFile]))
        with open(logFileName+"/"+csvFiles[iCsvFile]) as csvFile:
            stats = csvFile.readlines()
        
        gpuTimes[iFile][iCsvFile] = float(stats[0].split(",")[1][:-1])
        cpuTimes[iFile][iCsvFile] = float(stats[1].split(",")[1][:-1])
        mIou_fr = float(stats[26].split(",")[1][:-1])
        mIou_oc = float(stats[27].split(",")[1][:-1])
        mIou_un = float(stats[28].split(",")[1][:-1])
        mIoUs[iFile][iCsvFile] = (mIou_fr + mIou_oc + mIou_un)/3
    best_mIoUs[iFile] = np.max(mIoUs[iFile])
    mCpuTimes[iFile] = np.mean(cpuTimes[iFile])
    modelSizes[iFile] = float(stats[2].split(",")[1][:-1])

# # viz the stats
# fig = plt.figure(figsize=(6.3,4))
# ax1 = fig.add_subplot(111)
# for iFile in [0,1,2,3,4,7]:
#     ax1.plot(modelSizes[iFile],np.max(mIoUs[iFile]),"bo-")
# for iFile in [5,6]:
#     ax1.plot(modelSizes[iFile],mIoUs[iFile],"go-")
# fig = plt.figure(figsize=(6.3,4))
# ax1 = fig.add_subplot(111)
# for iFile in [0,1,2,3,4,7]:
#     ax1.plot([np.mean(cpuTimes[iFile])]*mIoUs[iFile].shape[0],mIoUs[iFile],"bo-")
# for iFile in [5,6]:
#     ax1.plot([np.mean(cpuTimes[iFile])]*mIoUs[iFile].shape[0],mIoUs[iFile],"go-")

# fig = plt.figure(figsize=(6.3,4))
# ax1 = fig.add_subplot(111)
# ax1.plot(modelSizes[[5,4]],best_mIoUs[[5,4]],"o-")
# ax1.plot(modelSizes[[0,1,2,3,4,7]],best_mIoUs[[0,1,2,3,4,7]],"o-")
# ax1.set_xlabel("model size [MB]")
# ax1.set_ylabel("mIoU [%]")

mCpuTimes = 1/mCpuTimes

trainTime = np.asarray([7.5, 5.75, 7.5, 6.25, 65.5, 115.0, 25.0, 18.75])  # [h]

fig1 = plt.figure(figsize=(5.5,4))
ax1 = fig1.add_subplot(111)
ax1.plot(mCpuTimes[[4,5,6]],best_mIoUs[[4,5,6]],"o-")
ax1.plot(mCpuTimes[[0,1,2,3,4,7]],best_mIoUs[[0,1,2,3,4,7]],"o-")

for row, idx in enumerate([0,1,2,3,4,7,5,6]):
    ax1.annotate(str(row), (mCpuTimes[idx]+0.2, best_mIoUs[idx]))
ax1.set_xlabel("mean CPU inference time [Hz]")
ax1.set_ylabel("mIoU [%]")

import tikzplotlib

tikzplotlib.save("./experiments/network_tuning/mIoU_network_tuning.tex",
                  axis_width="3.in", axis_height="3.0in",
                  strict=True, 
                  extra_axis_parameters=["scaled ticks=false",
                                         "tick label style={/pgf/number format/fixed}"])

fig2 = plt.figure(figsize=(5.5,4))
ax2 = fig2.add_subplot(111)
ax2.plot([4,6,7],trainTime[[4,6,7]],"o-")
ax2.plot([0,1,2,3,4,5],trainTime[:6],"o-")
ax2.set_xlabel("experiment number")
ax2.set_ylabel("train time until convergence [h]")

# ax1.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

tikzplotlib.save("./experiments/network_tuning/train_time_network_tuning.tex",
                  axis_width="2.75in", axis_height="3.0in",
                  strict=True, 
                  extra_axis_parameters=["scaled ticks=false",
                                         "tick label style={/pgf/number format/fixed}"])