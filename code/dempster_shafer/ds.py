import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.close('all')


def plot_ds_over_time(m, title, plot_x_axis=True):
    time = range(m.shape[1])
    plt.plot(time, m[0, :], "r")
    plt.plot(time, m[1, :], "g")
    plt.plot(time, m[2, :], "b")

    plt.ylim([-0.1, 1.1])
    plt.xlim([0., m.shape[1]])
    plt.ylabel(title)
    if not plot_x_axis:
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)


def limit_certainty(m, u_min):
    # unknown mass cannot fall beneath u_min
    m[:-1] = (1 - u_min) * m[:-1]
    m[2] = 1 - m[0] - m[1]
    return m


def fuse_masses(m1, m2, comb_rule=0, u_min=0.0, entropy_scaling=False, eps=1e-8, converg_factor=1.):
    """
    Fuses two masses defined as m = [{fr},{oc},{fr,oc}] according to the
    Dempster's Rule of Combination.

    combRule:
        0 = Yager
        1 = Yager with konflict being assigned to free and occ equally
        2 = Dempster
    """
    if entropy_scaling:
        # FIRST: rescale mass to have at least uMin unknown mass        
        m = limit_certainty(m1, u_min)

        # SECOND: rescale mass to account for redundant information        
        # current conflict 
        k = m1[0] * m2[1] + m1[1] * m2[0]
        # difference in information 
        h1_2 = np.clip(converg_factor * (m2[2] - m1[2]), 0., 1.)

        # limit the difference in information
        h1_2_max = np.clip((u_min - m2[2]) / (m2[2] * m1[2] - m2[2] + k), 0., 1.)

        if (m2[2] * m1[2] - m2[2] + k) < 0:
            h1_2 = np.min((h1_2, h1_2_max))
        else:
            # h1_2 = np.max((h1_2, h1_2_max))
            h1_2 = h1_2
        if h1_2 < 0.001:
            h1_2 = 0

        # scale fr and occ masses
        m1[:-1] *= h1_2
        m1[2] = 1. - np.sum(m1[:-1])

    # limit certainty (needed for Dempster comb rule)
    m2 = limit_certainty(m2, eps)

    # compute the conflict
    k = m1[0] * m2[1] + m1[1] * m2[0]

    # compute the fused masses of each class
    m = np.zeros((3, 1))

    # Yager
    if comb_rule == 0:
        m[0] = (m1[0] * m2[0] + m1[0] * m2[2] + m1[2] * m2[0])
        m[1] = (m1[1] * m2[1] + m1[1] * m2[2] + m1[2] * m2[1])
        m[2] = m1[2] * m2[2] + k
    # modified Yager (conflict equally assigned to fr & occ classes)
    elif comb_rule == 1:
        m[0] = m1[0] * m2[0] + m1[0] * m2[2] + m1[2] * m2[0] + k / 2
        m[1] = m1[1] * m2[1] + m1[1] * m2[2] + m1[2] * m2[1] + k / 2
        m[2] = m1[2] * m2[2]
    # Dempster
    else:
        m[0] = (m1[0] * m2[0] + m1[0] * m2[2] + m1[2] * m2[0]) / (1 - k)
        m[1] = (m1[1] * m2[1] + m1[1] * m2[2] + m1[2] * m2[1]) / (1 - k)
        m[2] = (m1[2] * m2[2]) / (1 - k)

    # normalize
    m[0] = np.clip(m[0], 0., 1.)
    m[1] = np.clip(m[1], 0., 1.)
    m[2] = 1 - m[0] - m[1]

    return m


# ============================MAIN=============================================#
# parameters
uMin = 0.2
ampFactor = 10.0

# define masses as m = [{fr},{oc},{fr,oc}] 
numMeas = 500
numMeasPart = int(numMeas // 10)
mMeas = np.zeros((3, numMeas))
mMeas[0, 0 * numMeasPart:1 * numMeasPart] = 0.0
mMeas[1, 0 * numMeasPart:1 * numMeasPart] = 0.4
mMeas[2, 0 * numMeasPart:1 * numMeasPart] = 0.6

mMeas[0, 1 * numMeasPart:2 * numMeasPart] = 0.0
mMeas[1, 1 * numMeasPart:2 * numMeasPart] = 0.7
mMeas[2, 1 * numMeasPart:2 * numMeasPart] = 0.3

mMeas[0, 2 * numMeasPart:3 * numMeasPart] = 1.0
mMeas[1, 2 * numMeasPart:3 * numMeasPart] = 0.0
mMeas[2, 2 * numMeasPart:3 * numMeasPart] = 0.0

mMeas[0, 3 * numMeasPart:4 * numMeasPart] = 0.1
mMeas[1, 3 * numMeasPart:4 * numMeasPart] = 0.5
mMeas[2, 3 * numMeasPart:4 * numMeasPart] = 0.4

mMeas[0, 4 * numMeasPart:5 * numMeasPart] = 0.9
mMeas[1, 4 * numMeasPart:5 * numMeasPart] = 0.0
mMeas[2, 4 * numMeasPart:5 * numMeasPart] = 0.1

mMeas[0, 5 * numMeasPart:6 * numMeasPart] = 0.0
mMeas[1, 5 * numMeasPart:6 * numMeasPart] = 0.9
mMeas[2, 5 * numMeasPart:6 * numMeasPart] = 0.1

mMeas[0, 6 * numMeasPart:] = 0.5
mMeas[1, 6 * numMeasPart:] = 0.5
mMeas[2, 6 * numMeasPart:] = 0.0

# mMeas[0,6*numMeasPart:7*numMeasPart] = 0.5
# mMeas[1,6*numMeasPart:7*numMeasPart] = 0.5
# mMeas[2,6*numMeasPart:7*numMeasPart] = 0.0
#
# mMeas[0,7*numMeasPart:8*numMeasPart] = 0.9
# mMeas[1,7*numMeasPart:8*numMeasPart] = 0.0
# mMeas[2,7*numMeasPart:8*numMeasPart] = 0.1
#
# mMeas[0,8*numMeasPart:9*numMeasPart] = 0.0
# mMeas[1,8*numMeasPart:9*numMeasPart] = 0.9
# mMeas[2,8*numMeasPart:9*numMeasPart] = 0.1
#
# mMeas[0,9*numMeasPart:10*numMeasPart] = 0.9
# mMeas[1,9*numMeasPart:10*numMeasPart] = 0.0
# mMeas[2,9*numMeasPart:10*numMeasPart] = 0.1

mUnDiff = np.zeros((3, numMeas))
mYager_ = np.zeros((3, numMeas))
mYager = np.zeros((3, numMeas))
mDempster = np.zeros((3, numMeas))
p1 = np.zeros((numMeas,))
p12 = np.zeros((numMeas,))
h1 = np.zeros((numMeas,))
h12 = np.zeros((numMeas,))
m0 = np.array([[0.], [0.], [1.]])

for it in range(numMeas):
    # fuse the fused signal with the last prediction
    if it == 0:
        mUnDiff[:, [it]] = m0
        mYager_[:, [it]] = m0
        mYager[:, [it]] = m0
        mDempster[:, [it]] = m0
    else:
        mUnDiff[:, [it]] = fuse_masses(mMeas[:, [it]].copy(), mUnDiff[:, it - 1], comb_rule=0, u_min=uMin,
                                       entropy_scaling=True, converg_factor=ampFactor)
        mYager_[:, [it]] = fuse_masses(mMeas[:, [it]].copy(), mYager_[:, it - 1], comb_rule=1)
        mYager[:, [it]] = fuse_masses(mMeas[:, [it]].copy(), mYager[:, it - 1], comb_rule=0)
        mDempster[:, [it]] = fuse_masses(mMeas[:, [it]].copy(), mDempster[:, it - 1], comb_rule=2)

print("")

numSubPlts = 5
subPltIdx = 0
fig = plt.figure()
subPltIdx += 1
plt.subplot(numSubPlts, 1, subPltIdx)
plot_ds_over_time(mMeas, "Input Masses", plot_x_axis=False)
plt.plot([0, numMeas], [uMin, uMin], "b--")
plt.legend(["free", "occupied", "unknown", "uMin"],
           loc="center", bbox_to_anchor=(0.5, 1.15), ncol=4)

subPltIdx += 1
plt.subplot(numSubPlts, 1, subPltIdx)
plot_ds_over_time(mUnDiff, "Unknown Diff", plot_x_axis=False)
plt.plot([0, numMeas], [uMin, uMin], "b--")

subPltIdx += 1
plt.subplot(numSubPlts, 1, subPltIdx)
plot_ds_over_time(mYager_, "with modified Yager", plot_x_axis=False)
plt.plot([0, numMeas], [uMin, uMin], "b--")

subPltIdx += 1
plt.subplot(numSubPlts, 1, subPltIdx)
plot_ds_over_time(mYager, "with Yager", plot_x_axis=False)
plt.plot([0, numMeas], [uMin, uMin], "b--")

subPltIdx += 1
plt.subplot(numSubPlts, 1, subPltIdx)
plot_ds_over_time(mDempster, "with Dempster", plot_x_axis=True)
plt.plot([0, numMeas], [uMin, uMin], "b--")
plt.xlabel("timestep")

storage_dir = Path("./").absolute().parents[1] / "imgs/08_occ_mapping_exp/choice_of_comb_rule/"
storage_dir = storage_dir / "ds_comb_rule_comparison.pgf"
fig_size = fig.get_size_inches()
fig_width_inches = 6.3
fig.set_size_inches(fig_width_inches, fig_width_inches * fig_size[0] / fig_size[1])
plt.savefig(str(storage_dir))

plt.show()
