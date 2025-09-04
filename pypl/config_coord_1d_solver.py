import numpy as np
from scipy import constants
from decimal import *

class config_coord_1d_solver:
    """
    One-dimensional configuration coordinate diagram solver.

    Parameters
    ----------
    omega_i : float
        Initial vibrational frequency (in meV).
    omega_f : float
        Final vibrational frequency (in meV).
    delta_q : float
        Displacement along the configuration coordinate (in Å·amu^{1/2}).
    prec : int, optional
        Precision for `decimal` operations, default is 100.
    """

    def __init__(self, omega_i, omega_f, delta_q, prec=100):

        self.omega_i = omega_i
        self.omega_f = omega_f
        self.delta_q = delta_q

        getcontext().prec = prec


    def compute_franck_condon_integrals(self, ni, nf):
        r"""
        Compute Franck-Condon (FC) overlap integrals between vibrational
        states in displaced harmonic oscillators.

        The integrals are defined as

        .. math::

            F_{ij} = \langle \chi_i(Q + \Delta Q) \mid \chi_j(Q) \rangle
                   = \langle \chi_i(Q) \mid \chi_j(Q - \Delta Q) \rangle

        with recurrence relations given in
        [Peder Thusgaard Ruhoff, Chem. Phys. 186, 355-374 (1994)].

        Parameters
        ----------
        ni : int
            Number of vibrational states in the initial potential.
        nf : int
            Number of vibrational states in the final potential.

        Notes
        -----
        `self.fc_ints`, a ndarray of shape (ni, nf), stores the Franck-Condon overlap integrals.
        """

        wi = self.omega_i * 1e-3 * constants.eV / constants.hbar**2
        wf = self.omega_f * 1e-3 * constants.eV / constants.hbar**2
        k = self.delta_q * constants.physical_constants["atomic mass constant"][0] ** 0.5 * 1e-10

        a = (wi - wf) / (wi + wf)
        b = 2 * k * np.sqrt(wi) * wf / (wi + wf)
        c = -a
        d = -2 * k * np.sqrt(wf) * wi / (wi + wf)
        e = 4 * np.sqrt(wi * wf) / (wi + wf)
        f = np.zeros((ni, nf))

        f[0, 0] = (e / 2) ** 0.5 * np.exp(b * d / 2 / e)

        f[0, 1] = 1 / np.sqrt(2) * d * f[0, 0]
        for j in range(2, nf):
            f[0, j] = 1 / np.sqrt(2 * j) * d * f[0, j - 1] + np.sqrt((j - 1) / j) * c * f[0, j - 2]

        f[1, 0] = 1 / np.sqrt(2) * b * f[0, 0]
        for i in range(2, ni):
            f[i, 0] = 1 / np.sqrt(2 * i) * b * f[i - 1, 0] + np.sqrt((i - 1) / i) * a * f[i - 2, 0]

        for j in range(1, nf):
            f[1, j] = 1 / np.sqrt(2 * 1) * b * f[0, j] + 1 / 2 * np.sqrt(j / 1) * e * f[0, j - 1]

        for i in range(2, ni):
            for j in range(1, nf):
                f[i, j] = 1 / np.sqrt(2 * i) * b * f[i - 1, j] + np.sqrt((i - 1) / i) * a * f[i - 2, j] + 1 / 2 * np.sqrt(j / i) * e * f[i - 1, j - 1]
        
        self.fc_ints = f
        self.ni = ni
        self.nf = nf


    @staticmethod
    def Gaussian(x, mu, sigma_r, sigma_factor=160):
        r"""
        Gaussian line shape function.

        .. math::

            G(x; \mu, \sigma) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( - \frac{(x - \mu)^2}{2\sigma^2} \right)

        where

        .. math::

            \sigma = ( \sigma_r[1] - \sigma_r[0] ) \cdot \frac{|\mu|}{\text{sigma\_factor}} + \sigma_r[0]

        Parameters
        ----------
        x : ndarray
            Energy axis.
        mu : float or ndarray
            Center position.
        sigma_r : tuple of float
            Range for Gaussian broadening (min, max).
        sigma_factor : float, optional
            Scaling factor for energy dependence of sigma, default is 160.

        Returns
        -------
        ndarray
            Gaussian line shape values.
        """

        sigma = (sigma_r[1] - sigma_r[0]) * abs(mu / sigma_factor) + sigma_r[0]
        prefactor = 1 / np.sqrt(2 * np.pi * sigma**2)
        exponent = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
        f = prefactor * exponent
        return f


    @staticmethod
    def Lorentzian(x, mu, gamma):
        r"""
        Lorentzian line shape function.

        .. math::

            L(x; \mu, \gamma) = \frac{1}{\pi \gamma} \frac{\gamma^2}{(x - \mu)^2 + \gamma^2}

        Parameters
        ----------
        x : ndarray
            Energy axis.
        mu : float
            Center position.
        gamma : float
            Broadening parameter.

        Returns
        -------
        ndarray
            Lorentzian line shape values.
        """

        pref = 1 / (np.pi * gamma)
        mp = gamma**2 / ((x - mu) ** 2 + gamma**2)
        f = pref * mp
        return f


    @staticmethod
    def BZ(ene, T):
        r"""
        Boltzmann factor.

        .. math::

            BZ(E, T) = \exp\!\left( -\frac{E}{k_B T} \right)

        Parameters
        ----------
        ene : float or ndarray
            Vibrational energy in meV.
        T : float
            Temperature in Kelvin.

        Returns
        -------
        float or ndarray
            Boltzmann factor.
        """

        bz_factor = np.exp(-(ene * 1e-3 * constants.eV) / (constants.k * T))
        return bz_factor


    def bulid_fc_lsp(self, eneaxis=None, temp=4, sigma=10, zpl_lorentzian=True, gamma=1):
        r"""
        Build the Franck-Condon (FC) lineshape function.

        The spectrum is given by

        .. math::

            I(E) = \sum_{i, j} p_i \, |F_{ij}|^2 \, G(E; E_{ij}, \sigma)

        where :math:`p_i` is the Boltzmann prefactor, :math:`F_{ij}` are
        Franck-Condon integrals, and :math:`G` is a Gaussian (or Lorentzian
        for the ZPL).

        Parameters
        ----------
        eneaxis : ndarray, optional
            Energy axis in meV. Default is `np.linspace(-1000, 1000, 2001)`.
        temp : float, optional
            Temperature in Kelvin. Default is 4 K.
        sigma : float, optional
            Gaussian broadening width. Default is 10 meV.
        zpl_lorentzian : bool, optional
            Replace zero-phonon line (ZPL) Gaussian with Lorentzian.
            Default is True.
        gamma : float, optional
            Lorentzian broadening parameter for ZPL. Default is 1 meV.

        Returns
        -------
        eneaxis : ndarray
            Energy axis (meV).
        fc_lineshape : ndarray
            Computed FC lineshape intensity.
        """

        if eneaxis is None:
            # meV
            eneaxis = np.linspace(-1000, 1000, 2001)

        self.temp = temp
        self.sigma = sigma
        self.zpl_lorentzian = zpl_lorentzian
        self.gamma = gamma

        self.energy_v = - self.omega_i * np.arange(self.ni)[:, None] + self.omega_f * np.arange(self.nf)[None, :]

        prefactors = self.BZ(self.omega_i * np.arange(self.ni), temp)
        # normalization
        prefactors /= (np.sum(prefactors))
        self.prefactors = prefactors

        lineshape = (self.fc_ints[:, :][:, :, None]**2 
                     * self.Gaussian(eneaxis[None, None, :], self.energy_v[:, :, None], sigma))
        lineshape = np.sum(prefactors[:, None, None] * lineshape, axis=(0, 1))
        if zpl_lorentzian:
            lineshape -= (self.fc_ints[0, 0]**2 * self.Gaussian(eneaxis, 0.0, sigma))
            lineshape += (self.fc_ints[0, 0]**2 * self.Lorentzian(eneaxis, 0.0, gamma))

        print('Integral check:', np.sum(lineshape) * (eneaxis[1] - eneaxis[0]))

        self.eneaxis = eneaxis
        self.fc_lineshape = lineshape

        return self.eneaxis, self.fc_lineshape


    def compute_spectrum(self, eneaxis=None, linshape=None, tdm=1.0, zpl=0.0, spectrum_type='pl'):
        r"""
        Compute the optical spectrum from the FC lineshape.

        For photoluminescence (PL):

        .. math::

            I_\text{PL}(E) \propto \text{linshape}(E) \cdot \mu^2 \cdot (E_\text{ZPL} - E)^3

        For absorption (ABS):

        .. math::

            I_\text{ABS}(E) \propto \text{linshape}(E) \cdot \mu^2 \cdot (E_\text{ZPL} + E)

        Parameters
        ----------
        eneaxis : ndarray, optional
            Energy axis (meV). Default is self.eneaxis.
        linshape : ndarray, optional
            Franck-Condon lineshape. Default is self.lineshape.
        tdm : float, optional
            Transition dipole moment. Default is 1.0.
        zpl : float, optional
            Zero-phonon line (ZPL) energy. Default is 0.0 meV.
        spectrum_type : {'pl', 'abs'}, optional
            Type of spectrum to compute: 'pl' (photoluminescence) or
            'abs' (absorption). Default is 'pl'.

        Returns
        -------
        eneaxis_out : ndarray
            Shifted energy axis (meV).
        spectrum : ndarray
            Spectrum intensity.
        """

        if eneaxis is None:
            eneaxis = self.eneaxis

        if linshape is None:
            linshape = self.fc_lineshape

        if spectrum_type == 'pl':
            eneaxis_out = zpl - eneaxis
            spectrum = linshape * tdm**2 * (zpl - eneaxis)**3
        elif spectrum_type == 'abs':
            eneaxis_out = zpl + eneaxis
            spectrum = linshape * tdm**2 * (zpl + eneaxis)
        else:
            raise ValueError("Error: wrong type. Must be 'pl' or 'abs'.")

        return eneaxis_out, spectrum


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    ##############
    # parameters #
    ##############

    # transition dipole moment: au
    tdm = 1.0

    # input freq (meV)
    # initial state. ES for PL
    freq_es = 65.53340
    # final state. GS for PL
    freq_gs = 59.08807

    # input q (amu^1/2 \AA)
    delta_q = 0.695740

    # order
    order_es = 50
    order_gs = 60

    # energy range for plot
    ene_range = [-200, 1200]

    # resolution
    resol = 1401

    # temperature: K
    temp = 5

    # broadening
    gamma = 0.3
    sigma = [8, 25]

    # ZPL: meV
    zpl = 1945

    ######################
    # compute pl and abs #
    ######################

    # photoluminescence
    ccd_pl = config_coord_1d_solver(freq_es, freq_gs, delta_q)

    ccd_pl.compute_franck_condon_integrals(ni=order_es, nf=order_gs)
    ccd_pl.bulid_fc_lsp(eneaxis=np.linspace(ene_range[0], ene_range[1], resol),
                        temp=temp, sigma=sigma, zpl_lorentzian=True, gamma=gamma)

    pl_spectrum = ccd_pl.compute_spectrum(tdm=tdm, zpl=zpl, spectrum_type='pl')

    # absorption
    ccd_abs = config_coord_1d_solver(freq_gs, freq_es, delta_q)

    ccd_abs.compute_franck_condon_integrals(ni=order_gs, nf=order_es)
    ccd_abs.bulid_fc_lsp(eneaxis=np.linspace(ene_range[0], ene_range[1], resol),
                         temp=temp, sigma=sigma, zpl_lorentzian=True, gamma=gamma)

    abs_spectrum = ccd_abs.compute_spectrum(tdm=tdm, zpl=zpl, spectrum_type='abs')

    ########
    # plot #
    ########

    fig, ax = plt.subplots(1, 1, figsize=(12, 4))

    red = "#DB4437"
    blue = "#4285F4"

    ax.plot(pl_spectrum[0] * 1e-3, pl_spectrum[1] / (np.sum(pl_spectrum[1]) * abs(pl_spectrum[0][1] - pl_spectrum[0][0])) * 1e3,
            color=red, linewidth=1, linestyle="-", label='PL')
    ax.plot(abs_spectrum[0] * 1e-3, abs_spectrum[1] / (np.sum(abs_spectrum[1]) * abs(abs_spectrum[0][1] - abs_spectrum[0][0])) * 1e3,
            color=blue, linewidth=1, linestyle="-", label='Abs')

    ax.set_xlim((1.5, 2.4))
    ax.set_ylim((0.0, 6))

    ax.legend(fontsize=12, loc="upper right", edgecolor="black")
    ax.grid(color='gray', linestyle='--', linewidth=0.5)

    ax.tick_params(direction="in")
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    ax.set_xlabel("$\hbar\omega$ (eV)")
    ax.set_ylabel("PL (arb. unit.)")

    plt.show()
