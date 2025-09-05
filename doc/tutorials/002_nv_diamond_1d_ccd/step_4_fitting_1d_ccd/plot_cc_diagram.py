#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12})
import sys
from matplotlib.cbook import get_sample_data
from scipy.optimize import curve_fit
from scipy import constants

##############
# Parameters #
##############

# mass weighted displacement delta Q
# unit is amu^{0.5} Å
dis = 0.695740

#######
# Fxn #
#######


def gs_fit_fun(x, a, c):
    return a * x**2 + c


def es_fit_fun(x, a, c):
    return a * (x - dis) ** 2 + c


###################
# Unit Conversion #
###################

eV_J = constants.eV
Ang_M = 1e-10
N_A = constants.N_A
AMU_kg = 1e-3 / N_A
hplanck = constants.h

#############
# Load data #
#############

fname = "gs_ene.dat"
gs_cods = np.loadtxt(fname, usecols=0, skiprows=0)
gs_ene = np.loadtxt(fname, usecols=1, skiprows=0)
ref_ene = np.min(gs_ene)
gs_ene = gs_ene - ref_ene
gs_ene = gs_ene * 13.6056980659

fname = "es_ene.dat"
es_cods = np.loadtxt(fname, usecols=0, skiprows=0)
es_ene = np.loadtxt(fname, usecols=1, skiprows=0)
es_ene = es_ene - ref_ene
es_ene = es_ene * 13.6056980659

gs_cods = gs_cods * dis
es_cods = es_cods * dis

#################
# Fit parameter #
#################

# GS
gs_params = curve_fit(gs_fit_fun, gs_cods[:5], gs_ene[:5])[0]
print("=" * 60)
print("Parameters for GS curve: ", gs_params)
gs_fit_ene = np.zeros(gs_ene.shape[0])
for j in range(gs_ene.shape[0]):
    gs_fit_ene[j] = gs_fit_fun(gs_cods[j], gs_params[0], gs_params[1])

gs_phonon = 2 * gs_params[0] * eV_J / (Ang_M**2 * AMU_kg)
gs_phonon = gs_phonon * (hplanck / 2 / np.pi) ** 2 / (eV_J**2)
gs_phonon = np.sqrt(gs_phonon) * 1000
print("GS phonon is %.5f meV" % gs_phonon)

# ES
es_params = curve_fit(es_fit_fun, es_cods[8:], es_ene[8:])[0]
print("=" * 60)
print("Parameetrs for ES curve: ", es_params)
es_fit_ene = np.zeros(es_ene.shape[0])
for j in range(es_ene.shape[0]):
    es_fit_ene[j] = es_fit_fun(es_cods[j], es_params[0], es_params[1])

es_phonon = 2 * es_params[0] * eV_J / (Ang_M**2 * AMU_kg)
es_phonon = es_phonon * (hplanck / 2 / np.pi) ** 2 / (eV_J**2)
es_phonon = np.sqrt(es_phonon) * 1000
print("ES phonon is %.5f meV" % es_phonon)

#######
# HRF #
#######

print("=" * 60)
gs_HRF = (
    (dis**2 * Ang_M**2 * AMU_kg)
    * (gs_phonon * eV_J * 1e-3 * 2 * np.pi / hplanck)
    / (2 * hplanck / 2 / np.pi)
)
print("HR for GS is %.5f" % gs_HRF)

es_HRF = (
    (dis**2 * Ang_M**2 * AMU_kg)
    * (es_phonon * eV_J * 1e-3 * 2 * np.pi / hplanck)
    / (2 * hplanck / 2 / np.pi)
)
print("HR for ES is %.5f" % es_HRF)

########
# Plot #
########

fig, ax = plt.subplots()

blue = "#4285F4"
red = "#DB4437"

ax.plot(
    es_cods,
    es_ene,
    color=blue,
    linestyle="",
    marker="s",
    markersize=5,
    label="ES energy",
)
ax.plot(
    gs_cods,
    gs_ene,
    color=red,
    linestyle="",
    marker="o",
    markersize=5,
    label="GS energy",
)

ax.plot(
    es_cods,
    es_fit_ene,
    color=blue,
    linestyle="--",
    linewidth=1.0,
    marker="",
    label="ES fit",
)
ax.plot(
    gs_cods,
    gs_fit_ene,
    color=red,
    linestyle="--",
    linewidth=1.0,
    marker="",
    label="GS fit",
)

ax.text(
    x=0.6,
    y=0.4,
    s="GS phonon: %.2f meV\nGS HRF: %.2f" % (gs_phonon, gs_HRF),
    color=red,
    transform=ax.transAxes,
)
ax.text(
    x=0.6,
    y=0.6,
    s="ES phonon: %.2f meV\nES HRF: %.2f" % (es_phonon, es_HRF),
    color=blue,
    transform=ax.transAxes,
)

ax.axvline(x=0.0, color="gray", linestyle="--", linewidth=1)
ax.axvline(x=dis, color="gray", linestyle="--", linewidth=1)

ax.legend(fontsize=12, loc="center left", edgecolor="black")
ax.set_xlim((-0.2, 0.9))
plt.xlabel("Q (amu$^{1/2}$ Å)")
plt.ylabel("Total Energy (eV)")
plt.tick_params(direction="in")

plt.savefig("CC_diagrams.pdf", bbox_inches="tight", dpi=300)
plt.show()
