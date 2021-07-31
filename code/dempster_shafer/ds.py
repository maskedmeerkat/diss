import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
from pathlib import Path

plt.close('all')


def plot_ds_over_time(m, title, plot_x_axis=True):
    time = range(m.shape[1])
    plt.plot(time, m[0, :], "r", label=r'free')
    plt.plot(time, m[1, :], "g", label=r'occupied')
    plt.plot(time, m[2, :], "b", label=r'unknown')

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


def fuse_masses(m1, m2, comb_rule=0, u_min=0.0, entropy_scaling=False, eps=1e-10, converg_factor=1.):
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
numMeasPart = 20
numMeas = numMeasPart * 8
mMeas = np.zeros((3, numMeas))
step = -1

step += 1
mMeas[0, step * numMeasPart:(step + 1) * numMeasPart] = 0.0
mMeas[1, step * numMeasPart:(step + 1) * numMeasPart] = 0.4
mMeas[2, step * numMeasPart:(step + 1) * numMeasPart] = 0.6

step += 1
mMeas[0, step * numMeasPart:(step + 1) * numMeasPart] = 0.0
mMeas[1, step * numMeasPart:(step + 1) * numMeasPart] = 0.5
mMeas[2, step * numMeasPart:(step + 1) * numMeasPart] = 0.5

step += 1
mMeas[0, step * numMeasPart:(step + 1) * numMeasPart] = 0.6
mMeas[1, step * numMeasPart:(step + 1) * numMeasPart] = 0.0
mMeas[2, step * numMeasPart:(step + 1) * numMeasPart] = 0.4

step += 1
mMeas[0, step * numMeasPart:(step + 1) * numMeasPart] = 0.8
mMeas[1, step * numMeasPart:(step + 1) * numMeasPart] = 0.0
mMeas[2, step * numMeasPart:(step + 1) * numMeasPart] = 0.2

step += 1
mMeas[0, step * numMeasPart:(step + 1) * numMeasPart] = 0.6
mMeas[1, step * numMeasPart:(step + 1) * numMeasPart] = 0.4
mMeas[2, step * numMeasPart:(step + 1) * numMeasPart] = 0.0

step += 1
mMeas[0, step * numMeasPart:(step + 1) * numMeasPart] = 1.0
mMeas[1, step * numMeasPart:(step + 1) * numMeasPart] = 0.0
mMeas[2, step * numMeasPart:(step + 1) * numMeasPart] = 0.0

step += 1
mMeas[0, step * numMeasPart:(step + 1) * numMeasPart] = 0.0
mMeas[1, step * numMeasPart:(step + 1) * numMeasPart] = 1.0
mMeas[2, step * numMeasPart:(step + 1) * numMeasPart] = 0.0

step += 1
mMeas[0, step * numMeasPart:] = 0.5
mMeas[1, step * numMeasPart:] = 0.5
mMeas[2, step * numMeasPart:] = 0.0

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
plt.plot([0, numMeas], [uMin, uMin], "b--", label=r"$\underline{m}_u$")
# plt.plot([0, numMeas], [uMin, uMin], "b--", label="m_u")
plt.legend(loc="center", bbox_to_anchor=(0.5, 1.15), ncol=4)

subPltIdx += 1
plt.subplot(numSubPlts, 1, subPltIdx)
plot_ds_over_time(mDempster, "Dempster", plot_x_axis=False)
plt.plot([0, numMeas], [uMin, uMin], "b--")

subPltIdx += 1
plt.subplot(numSubPlts, 1, subPltIdx)
plot_ds_over_time(mYager, "Yager", plot_x_axis=False)
plt.plot([0, numMeas], [uMin, uMin], "b--")

subPltIdx += 1
plt.subplot(numSubPlts, 1, subPltIdx)
plot_ds_over_time(mYager_, "YaDer", plot_x_axis=False)
plt.plot([0, numMeas], [uMin, uMin], "b--")

subPltIdx += 1
last_ax = fig.add_subplot(numSubPlts, 1, subPltIdx)
plot_ds_over_time(mUnDiff, "lower-bounded Yager", plot_x_axis=True)
plt.plot([0, numMeas], [uMin, uMin], "b--")

# label the time axis
time = range(mUnDiff.shape[1])
last_ax.set_xticks(np.arange(9) * numMeasPart)
plt.xlabel("timestep")

# remove white space margin around figure
plt.tight_layout()

storage_dir = Path("./").absolute().parents[1] / "imgs/08_occ_mapping_exp/choice_of_comb_rule/"
storage_dir = storage_dir / "ds_comb_rule_comparison.pgf"
fig_size = fig.get_size_inches()
fig_width_inches = 6.3
fig.set_size_inches(fig_width_inches, fig_width_inches * fig_size[0] / fig_size[1])
plt.savefig(str(storage_dir))

plt.show()
