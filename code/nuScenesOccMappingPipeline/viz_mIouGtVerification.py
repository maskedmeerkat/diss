import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# plt.style.use("ggplot")
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
plt.close("all")

interUnionPx = np.load("./experiments/gtVerification/mIouGtVerification.npy")
mIoU_occ = interUnionPx[:,1,0]/interUnionPx[:,1,1]*100

heightThresholds = np.array([0.025,0.05,0.075,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0])

mIoU_occ_heights = mIoU_occ[:len(heightThresholds)]
mIoU_occ_semseg = mIoU_occ[len(heightThresholds):]

fig = plt.figure(figsize=(6.3,4))
ax1 = fig.add_subplot(111)
ax1.plot(heightThresholds,mIoU_occ_heights,"bo-")

xlim = ax1.get_xlim()
for i in range(mIoU_occ_semseg.shape[0]):
    if not(i==2):
        ax1.plot(xlim,[mIoU_occ_semseg[i]]*2,"--")

# axes
# axes = plt.gca()
ylim = ax1.get_ylim()
ax1.plot([heightThresholds[7],heightThresholds[7]],[ylim[0],mIoU_occ[7]],linestyle="--",color="b")
ax1.plot([xlim[0],heightThresholds[7]],[mIoU_occ[7],mIoU_occ[7]],linestyle="--",color="b")
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
ax1.set_xlabel("height thresholds [m]")
ax1.set_ylabel("mIoU of occupied pixels [%]")
ax1.legend(['(best mIoU {:.2f}) height thresholds'.format(np.max(mIoU_occ_heights)),
            "(mIoU {:.2f}) no street".format(mIoU_occ_semseg[0]),
            "(mIoU {:.2f}) no street or sidewalk".format(mIoU_occ_semseg[1]),
            "(mIoU {:.2f}) no street, sidwalk or terrain".format(mIoU_occ_semseg[3])])
plt.show()
# import tikzplotlib
#
# tikzplotlib.save("./experiments/gtVerification/mIouGtVerification.tex",
#                  axis_width="5.5in", axis_height="3.0in",
#                  strict=True, extra_axis_parameters=["scaled ticks=false", "tick label style={/pgf/number format/fixed}"])