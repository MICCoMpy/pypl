#!/usr/bin/env python3

import numpy as np
import sys
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.integrate import quad
import time


def Gaussian(x, mu, sigma):
    prefix = 1 / np.sqrt(2 * np.pi * sigma**2)
    eeee = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return prefix * eeee


filename00 = "10_pl_1d_ccd.dat"
CCDene = np.loadtxt(filename00, usecols=0, skiprows=1)
CCDlsp = np.loadtxt(filename00, usecols=1, skiprows=1)

filename20 = "HSE-allmodes-512.dat"

filename30 = "HSE-allmodes-extrapolated.dat"
diluteene = np.loadtxt(filename30, usecols=0, skiprows=0)
dilutelsp = np.loadtxt(filename30, usecols=1, skiprows=0)
dilutelsp[:] = dilutelsp[:] * (1945 - diluteene[:]) ** 3


ENEAXIS = np.loadtxt(filename20, usecols=0)

DATA = np.zeros((5, ENEAXIS.shape[0]))
DATA[3] = np.loadtxt(filename20, usecols=1)

#######
for j in range(5):
    for k in range(ENEAXIS.shape[0]):
        DATA[j, k] = DATA[j, k] * (1945 - ENEAXIS[k]) ** 3
######

fig, ax = plt.subplots(1, 1, figsize=(6, 4))

colors = ["#4285F4", "#DB4437", "#F4B400", "#0F9D58"]
linewidths = [1.5, 1.5, 1.5, 1.5, 1.5]
linestyles = ["-", "-", "-", "-", "-"]
labels = ["1D CCD", "All modes", "All modes, dilute limit"]

ax.plot(
    CCDene,
    CCDlsp / max(CCDlsp) * 0.48,
    color=colors[1],
    label=labels[0],
    linewidth=linewidths[0],
    linestyle=linestyles[0],
)

for j in range(5):
    if j in [3]:
        ax.plot(
            -ENEAXIS,
            2 * DATA[j] / max(DATA[j]),
            color=colors[0],
            label=labels[1],
            linewidth=linewidths[j],
            linestyle=linestyles[j],
        )

ax.plot(
    -diluteene,
    4 * dilutelsp / max(dilutelsp),
    color=colors[3],
    label=labels[2],
    linewidth=linewidths[0],
    linestyle=linestyles[0],
)


#######
# Exp
fname = "NV-PL-exp.csv"
f = open(fname, "r")
NOP = 186
Exp_DATA = np.zeros((2, NOP))
f.readline()
for i in range(NOP):
    line = f.readline()
    Exp_DATA[0][i] = float(line.split(",")[0])
    Exp_DATA[1][i] = float(line.split(",")[1])
f.close()

Exp_DATA[0] = Exp_DATA[0] - 1945

ax.fill_between(
    Exp_DATA[0],
    0,
    0.40 * Exp_DATA[1] / max(Exp_DATA[1]),
    color="grey",
    label="Expt.",
    alpha=0.4,
)
#######

ax.set_xlim((-550, 10))
ax.set_ylim((0, 0.4))
ax.set_yticklabels([])
ax.tick_params(direction="in")
ax.legend(
    fontsize=12,
    loc="upper left",
    edgecolor="black",
)
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")
ax.set_xlabel("$\hbar\omega - E_{\mathrm{ZPL}}$ (meV)")
ax.set_ylabel("PL Intensity (arb. unit)")

plt.savefig("figure_1D_CCD_all_modes_dilute.pdf", dpi=300, bbox_inches="tight")
plt.show()
