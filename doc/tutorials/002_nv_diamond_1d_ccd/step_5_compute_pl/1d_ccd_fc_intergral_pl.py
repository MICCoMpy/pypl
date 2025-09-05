#!/usr/bin/env python3
import numpy as np
from scipy import constants
from scipy.special import eval_hermite
from scipy.special import comb, factorial, factorial2
import sys
import matplotlib.pyplot as plt
from decimal import *

plt.rcParams.update({"font.size": 12})

getcontext().prec = 150

#############
# Functions #
#############


# fc integral
def fcf_mine(ni, nf, wi, wf, k):
    hbar = Decimal(constants.hbar)
    eV = Decimal(constants.eV)
    amu = Decimal(constants.physical_constants["atomic mass constant"][0])
    ang = Decimal(1e-10)
    k = Decimal(k)
    wi = Decimal(wi)
    wi = wi * eV  # J
    wi = wi / hbar  # rad \cdot s^{-1}
    wi = wi / hbar  # J^{-1} \cdot rad \cdot s^{-2}
    wf = Decimal(wf)
    wf = wf * eV  # J
    wf = wf / hbar  # rad \cdot s^{-1}
    wf = wf / hbar  # J^{-1} \cdot rad \cdot s^{-2}
    # kg^{0.5} \cdot meter
    k = k * np.sqrt(amu) * ang

    a = (wi - wf) / (wi + wf)
    b = 2 * k * np.sqrt(wi) * wf / (wi + wf)
    c = -a
    d = -2 * k * np.sqrt(wf) * wi / (wi + wf)
    e = 4 * np.sqrt(wi * wf) / (wi + wf)
    f = np.zeros((ni, nf), dtype=Decimal)

    f[0, 0] = np.sqrt(e / 2) * np.exp(b * d / 2 / e)

    f[0, 1] = 1 / np.sqrt(Decimal(2)) * d * f[0, 0]
    for j in range(2, nf):
        f[0, j] = (
            1 / np.sqrt(Decimal(2 * j)) * d * f[0, j - 1]
            + np.sqrt(Decimal(j - 1) / Decimal(j)) * c * f[0, j - 2]
        )

    f[1, 0] = 1 / np.sqrt(Decimal(2)) * b * f[0, 0]
    for i in range(2, ni):
        f[i, 0] = (
            1 / np.sqrt(Decimal(2 * i)) * b * f[i - 1, 0]
            + np.sqrt(Decimal(i - 1) / Decimal(i)) * a * f[i - 2, 0]
        )

    for j in range(1, nf):
        f[1, j] = (
            1 / np.sqrt(Decimal(2 * 1)) * b * f[0, j]
            + 1 / Decimal(2) * np.sqrt(Decimal(j / 1)) * e * f[0, j - 1]
        )

    for i in range(2, ni):
        for j in range(1, nf):
            f[i, j] = (
                1 / np.sqrt(Decimal(2 * i)) * b * f[i - 1, j]
                + np.sqrt(Decimal(i - 1) / Decimal(i)) * a * f[i - 2, j]
                + 1
                / Decimal(2)
                * np.sqrt(Decimal(j) / Decimal(i))
                * e
                * f[i - 1, j - 1]
            )
    return f.astype(float)


# gaussian function
def Gaussian(x, mu, sigma):
    pref = 1 / np.sqrt(2 * np.pi * sigma**2)
    expp = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return pref * expp


# lorentzian function
def Lorentzian(x, mu, gamma):
    pref = 1 / (np.pi * gamma)
    mp = gamma**2 / ((x - mu) ** 2 + gamma**2)
    return pref * mp


# BZ distribution
def BZ(ene, T):  # unit is meV and K
    return np.exp(-(ene * 1e-3 * constants.eV) / (constants.k * T))


# Temp factor
def fcf_temp(order_a, order_b, freq_a, fc_int_v, energy_v, Temp):
    fc_int_sq_temp = np.zeros((order_a, order_b))
    for ind_a in range(order_a):
        pref = BZ(ind_a * freq_a, Temp)
        fc_int_sq_temp[ind_a, :] = pref * abs(fc_int_v[ind_a, :]) ** 2
    return fc_int_sq_temp


# compute line shape
def bulid_lsp(eneaxis, order_a, order_b, fc_int_sq_temp, energy_v, sigma):
    lsp = np.zeros(eneaxis.shape[0])
    for ind_a in range(order_a):
        for ind_b in range(order_b):
            lsp[:] = lsp[:] + fc_int_sq_temp[ind_a, ind_b] * Gaussian(
                eneaxis[:], energy_v[ind_a, ind_b], sigma
            )
    return lsp


# compute line shape v2
def bulid_lsp_v2(eneaxis, order_a, order_b, fc_int_sq_temp, energy_v, sigma, gamma):
    lsp = np.zeros(eneaxis.shape[0])
    for ind_a in range(order_a):
        for ind_b in range(order_b):
            if ind_a == 0 and ind_b == 0:
                lsp[:] = lsp[:] + fc_int_sq_temp[ind_a, ind_b] * Lorentzian(
                    eneaxis[:], energy_v[ind_a, ind_b], gamma
                )
            else:
                lsp[:] = lsp[:] + fc_int_sq_temp[ind_a, ind_b] * Gaussian(
                    eneaxis[:], energy_v[ind_a, ind_b], sigma
                )
    return lsp


if __name__ == "__main__":

    #########
    # Input #
    #########

    # Note: a stands for the initial state and b stands for the final state
    # For photoluminescence, a stands for the excited state and b stands for
    # the ground state
    # input freq (meV)
    freq_a = 65.53340
    freq_b = 59.08807

    # input q (amu^1/2 \AA)
    raw_q = 0.695740

    # order
    order_a = 100
    order_b = 100

    # energy range for plot
    ene_range = [-1000, 500]

    # resolution
    resol = 1501

    # temperature: K
    Temp = 10

    # broadening
    gamma = 2
    sigma = 25

    # ZPL: meV
    ZPL = 1945

    ################
    # FC Integrals #
    ################

    # case 1: using frequencies for the initial and final states

    # unit conversion, from meV to eV
    alpha_a = freq_a * 1e-3
    alpha_b = freq_b * 1e-3

    # compute fc integrals
    fc_int_v = fcf_mine(order_a, order_b, alpha_a, alpha_b, raw_q)

    ###################
    # TA FC Integrals #
    ###################

    # case 2: using only the frequencies for the ground state

    # unit conversion, from meV to eV
    alpha_a = freq_a * 1e-3
    alpha_b = alpha_a

    # compute fc integrals
    ta_fc_int_v = fcf_mine(order_a, order_b, alpha_a, alpha_b, raw_q)

    ###################
    # TB FC Integrals #
    ###################

    # case 3: using only the frequencies for the excited state

    # unit conversion, from meV to eV
    alpha_b = freq_b * 1e-3
    alpha_a = alpha_b

    # compute fc integrals
    tb_fc_int_v = fcf_mine(order_a, order_b, alpha_a, alpha_b, raw_q)

    ##############
    # line shape #
    ##############

    # compute the energies corresponding to fc intergrals
    energy_v = np.zeros((order_a, order_b))
    for ind_a in range(order_a):
        for ind_b in range(order_b):
            energy_v[ind_a, ind_b] = 0 + ind_a * freq_a - ind_b * freq_b

    ta_energy_v = np.zeros((order_a, order_b))
    for ind_a in range(order_a):
        for ind_b in range(order_b):
            ta_energy_v[ind_a, ind_b] = 0 + ind_a * freq_a - ind_b * freq_a

    tb_energy_v = np.zeros((order_a, order_b))
    for ind_a in range(order_a):
        for ind_b in range(order_b):
            tb_energy_v[ind_a, ind_b] = 0 + ind_a * freq_b - ind_b * freq_b

    # line shape
    eneaxis = np.linspace(ene_range[0], ene_range[1], resol)
    lsp = np.zeros(resol)
    ta_lsp = np.zeros(resol)
    tb_lsp = np.zeros(resol)

    fc_int_sq_temp = fcf_temp(order_a, order_b, freq_a, fc_int_v, energy_v, Temp)
    # lsp = bulid_lsp(eneaxis, order_a, order_b, fc_int_sq_temp, energy_v, sigma)
    lsp = bulid_lsp_v2(
        eneaxis, order_a, order_b, fc_int_sq_temp, energy_v, sigma, gamma
    )

    ta_fc_int_sq_temp = fcf_temp(order_a, order_b, freq_a, ta_fc_int_v, energy_v, Temp)
    # ta_lsp = bulid_lsp(eneaxis, order_a, order_b, ta_fc_int_sq_temp, ta_energy_v, sigma)
    ta_lsp = bulid_lsp_v2(
        eneaxis, order_a, order_b, ta_fc_int_sq_temp, energy_v, sigma, gamma
    )

    tb_fc_int_sq_temp = fcf_temp(order_a, order_b, freq_b, tb_fc_int_v, energy_v, Temp)
    # tb_lsp = bulid_lsp(eneaxis, order_a, order_b, tb_fc_int_sq_temp, tb_energy_v, sigma)
    tb_lsp = bulid_lsp_v2(
        eneaxis, order_a, order_b, tb_fc_int_sq_temp, energy_v, sigma, gamma
    )

    # pre-factor
    lsp[:] = lsp[:] * (ZPL + eneaxis[:]) ** 3
    norm = sum(lsp) * (ene_range[1] - ene_range[0]) / resol
    lsp[:] = lsp[:] / norm

    ta_lsp[:] = ta_lsp[:] * (ZPL + eneaxis[:]) ** 3
    norm = sum(ta_lsp) * (ene_range[1] - ene_range[0]) / resol
    ta_lsp[:] = ta_lsp[:] / norm

    tb_lsp[:] = tb_lsp[:] * (ZPL + eneaxis[:]) ** 3
    norm = sum(tb_lsp) * (ene_range[1] - ene_range[0]) / resol
    tb_lsp[:] = tb_lsp[:] / norm

    ########
    # Plot #
    ########

    fig, ax = plt.subplots(1, 1)

    blue = "#4285F4"
    red = "#DB4437"
    green = "#0F9D58"

    ax.plot(
        eneaxis,
        lsp,
        color=red,
        linewidth=2,
        linestyle="-",
        label="$\omega_g=$ %.2f meV, $\omega_e=$ %.2f meV" % (freq_b, freq_a),
    )
    ax.plot(
        eneaxis,
        ta_lsp,
        color=blue,
        linewidth=2,
        linestyle="-",
        label="$\omega_e=$ %.2f meV" % freq_a,
    )
    ax.plot(
        eneaxis,
        tb_lsp,
        color=green,
        linewidth=2,
        linestyle="-",
        label="$\omega_g=$ %.2f meV" % freq_b,
    )

    fname = str(Temp) + "_pl_1d_ccd.dat"
    np.savetxt(fname, np.c_[eneaxis, lsp], fmt="%10.6e")

    ax.legend(fontsize=12, loc="upper left", edgecolor="black")
    ax.set_xlim((-500, 100))
    ax.set_ylim((0, 0.01))
    ax.tick_params(direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.set_xlabel("Energy (meV)")
    ax.set_ylabel("PL Intensity (arb. unit)")

    plt.savefig(str(Temp) + "_pl_1d_ccd.pdf", dpi=300, bbox_inches="tight")
    plt.show()
