import numpy as np
from scipy import constants
import warnings


class hrf:
    """
    Compute Huang-Rhys factors and spectral density from phonon
    frequencies, eigenmodes, and either forces or displacements.
    All inputs, except atomic masses, should be in SI units.

    Parameters
    ----------
    freqs : ndarray of shape (3N,)
        Phonon frequencies in rad/s.
    modes : ndarray of shape (3N, N, 3)
        Phonon eigenmodes.
    atomic_symbols : list of str
        Atomic chemical symbols.
    """

    def __init__(self, freqs, modes, atomic_symbols):

        self.freqs = freqs
        self.modes = modes
        self.atomic_symbols = atomic_symbols

        self.nom = freqs.shape[0]
        self.nom_translational = 3

        self.nom_imag_freq = np.sum(np.where(self.freqs[3:] < 0.0))
        if self.nom_imag_freq > 0:
            warnings.warn(f"{self.nom_imag_freq} imaginary mode(s) detected in the phonon spectrum. "
                        "Results involving these modes (e.g., HR factors) may be unreliable.",
                        UserWarning)


    def set_masses(self, mass_list=None):
        """
        Set atomic masses in kilograms for all atoms in the system.

        Parameters
        ----------
        mass_list : dict, optional
            Dictionary mapping element symbols to atomic masses in unified atomic
            mass units (u). If not provided, IUPAC-recommended values are used.

        Notes
        -----
        The converted atomic masses are stored in ``self.masses`` in kilograms.
        """

        default_mass_list = mass_list = {
            "H": 1.00784, "He": 4.002602,
            "Li": 6.938, "Be": 9.0121831, "B": 10.806, "C": 12.0096, "N": 14.00643, "O": 15.99903, "F": 18.99840316, "Ne": 20.1797,
            "Na": 22.98976928, "Mg": 24.304, "Al": 26.9815385, "Si": 28.084, "P": 30.973761998, "S": 32.059, "Cl": 35.446, "Ar": 39.948,
            "K": 39.0983, "Ca": 40.078, "Sc": 44.955908, "Ti": 47.867, "V": 50.9415, "Cr": 51.9961, "Mn": 54.938043, "Fe": 55.845,
            "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723, "Ge": 72.63, "As": 74.921595, "Se": 78.971,
            "Br": 79.901, "Kr": 83.798, "Rb": 85.4678, "Sr": 87.62, "Y": 88.90584, "Zr": 91.224, "Nb": 92.90637, "Mo": 95.95,
            "Tc": 98.0, "Ru": 101.07, "Rh": 102.90550, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414, "In": 114.818, "Sn": 118.710,
            "Sb": 121.760, "Te": 127.60, "I": 126.90447, "Xe": 131.293, "Cs": 132.90545196, "Ba": 137.327,
            "La": 138.90547, "Ce": 140.116, "Pr": 140.90766, "Nd": 144.242, "Pm": 145.0, "Sm": 150.36, "Eu": 151.964,
            "Gd": 157.25, "Tb": 158.92535, "Dy": 162.500, "Ho": 164.93033, "Er": 167.259, "Tm": 168.93422, "Yb": 173.045,
            "Lu": 174.9668, "Hf": 178.49, "Ta": 180.94788, "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217,
            "Pt": 195.084, "Au": 196.966569, "Hg": 200.592, "Tl": 204.38, "Pb": 207.2, "Bi": 208.98040,
            "Po": 209.0, "At": 210.0, "Rn": 222.0, "Fr": 223.0, "Ra": 226.0,
            "Ac": 227.0, "Th": 232.0377, "Pa": 231.03588, "U": 238.02891,
            "Np": 237.0, "Pu": 244.0, "Am": 243.0, "Cm": 247.0, "Bk": 247.0,
            "Cf": 251.0, "Es": 252.0, "Fm": 257.0, "Md": 258.0, "No": 259.0,
            "Lr": 262.0, "Rf": 267.0, "Db": 270.0, "Sg": 271.0, "Bh": 270.0,
            "Hs": 277.0, "Mt": 278.0, "Ds": 281.0, "Rg": 282.0, "Cn": 285.0,
            "Nh": 286.0, "Fl": 289.0, "Mc": 290.0, "Lv": 293.0, "Ts": 294.0, "Og": 294.0
        }

        if mass_list is None:
            print("Using IUPAC-recommended atomic masses (as provided by ChatGPT).")
            mass_list = default_mass_list

        missing = [sym for sym in self.atomic_symbols if sym not in mass_list]
        if missing:
            raise ValueError(f"Missing atomic masses for the following elements: {', '.join(missing)}")
        
        self.masses = np.array([mass_list[sym] for sym in self.atomic_symbols])
        self.masses *= constants.physical_constants['atomic mass constant'][0]


    def compute_hrf_forces(self, forces):
        r"""
        Compute Huang-Rhys factors using atomic forces.

        .. math::

            S_k = \frac{\omega_k \Delta Q_k^2}{2 \hbar}

        where

        .. math::

            \Delta Q_k = \frac{1}{\omega_k^2} \sum_{\alpha=1}^N \sum_{i=x,y,z} \frac{F_{\alpha i}}{\sqrt{M_\alpha}} e_{k, \alpha i}

        Parameters
        ----------
        forces : ndarray of shape (N, 3)
            Forces on atoms in J/m (SI units).

        Notes
        -----
        The computed Huang-Rhys factors are stored in ``self.hrf``.
        """

        mass_forces = forces * np.power(self.masses, -0.5)[:, None]
        mass_forces = mass_forces.ravel()

        modes = np.reshape(self.modes, (self.nom, self.nom))
        deltaq = np.dot(modes[self.nom_translational + self.nom_imag_freq:, :], mass_forces)
        deltaq = deltaq / self.freqs[self.nom_translational + self.nom_imag_freq:]**2

        self.hrf = np.concatenate((np.zeros(self.nom_translational + self.nom_imag_freq, dtype=np.float64),
                                   deltaq**2 * self.freqs[self.nom_translational + self.nom_imag_freq:] / 2 / constants.hbar))


    def compute_hrf_dis(self, gs_coord, es_coord, cell_parameters):
        r"""
        Compute Huang-Rhys factors using atomic displacements.

        .. math::

            S_k = \frac{\omega_k \Delta Q_k^2}{2 \hbar}

        with

        .. math::

            \Delta Q_k = \sum_{\alpha=1}^N \sum_{i=x,y,z} \sqrt{M_\alpha} \Delta R_{\alpha i} e_{k, \alpha i}

        Parameters
        ----------
        gs_coord : ndarray of shape (N, 3)
            Ground-state atomic coordinates (in meters).
        es_coord : ndarray of shape (N, 3)
            Excited-state atomic coordinates (in meters).
        cell_parameters : ndarray of shape (3, 3)
            Lattice vectors of the simulation cell (in meters).

        Notes
        -----
        Displacements are wrapped back into the unit cell.
        The computed Huang-Rhys factors are stored in ``self.hrf``.
        """

        inv_cell = np.linalg.inv(cell_parameters.T)
        gs_frac = np.dot(gs_coord, inv_cell)
        es_frac = np.dot(es_coord, inv_cell)
        dis_frac = es_frac - gs_frac
        dis_frac -= np.round(dis_frac)
        dis_cart = np.dot(dis_frac, cell_parameters)

        self.dis = dis_cart

        mass_dis = self.dis * np.power(self.masses, 0.5)[:, None]
        mass_dis = mass_dis.flatten()

        modes = np.reshape(self.modes, (self.nom, self.nom))
        deltaq = np.dot(modes[self.nom_translational + self.nom_imag_freq:, :], mass_dis)

        self.hrf = np.concatenate((np.zeros(self.nom_translational + self.nom_imag_freq, dtype=np.float64),
                                   deltaq**2 * self.freqs[self.nom_translational + self.nom_imag_freq:] / 2 / constants.hbar))


    @staticmethod
    def gaussian(x, mu, sigma):
        r"""
        Compute the value of a Gaussian function.

        .. math::

            G(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)

        Parameters
        ----------
        x : float
            Evaluation point.
        mu : float
            Mean of the Gaussian.
        sigma : float
            Standard deviation.

        Returns
        -------
        float
            Value of the Gaussian at `x`.
        """

        prefactor = 1 / np.sqrt(2.0 * np.pi * sigma**2)
        exponent = np.exp(- (x - mu)**2 / (2 * sigma**2))
        f = prefactor * exponent
        return f


    @staticmethod
    def f_sigma(freqs, sigma):
        r"""
        Compute a frequency-dependent linewidth function by linear interpolation.

        .. math::

            \sigma(\omega_k) = \sigma_0 - \frac{\sigma_0 - \sigma_1}{\max(\omega_k) - \min(\omega_k)} (\omega_k - \min(\omega_k))

        Parameters
        ----------
        freqs : ndarray of shape (M,)
            Phonon frequencies in rad/s.
        sigma : list of float
            Two-element list ``[sigma_0, sigma_1]`` in meV, giving the linewidth
            at the minimum and maximum frequency.

        Returns
        -------
        collect_sigma : ndarray of shape (M,)
            Interpolated linewidths :math:`\sigma(\omega_k)` in meV.
        """

        collect_sigma = sigma[0] - (sigma[0] - sigma[1]) / (max(freqs) - min(freqs)) * (freqs - min(freqs))
        return collect_sigma


    def compute_spectral_density(self, energy_axis, sigma):
        r"""
        Compute the phonon spectral density with Gaussian broadening.

        .. math::

            S(\hbar \omega) = \sum_k S_k G(\hbar \omega, \hbar \omega_k, \sigma(\omega_k))

        where :math:`G` is a normalized Gaussian.

        Parameters
        ----------
        energy_axis : ndarray of shape (N,)
            Energy axis in meV over which to evaluate the spectral density.
        sigma : list of float
            Two-element list ``[sigma_0, sigma_1]`` in meV, used for linewidth interpolation.

        Returns
        -------
        spectral_density : ndarray of shape (N,)
            Spectral density evaluated on the given energy axis (in 1/meV).
        """

        gfxns = np.zeros((self.nom, energy_axis.shape[0]))

        collect_sigmas = self.f_sigma(self.freqs, sigma)

        gfxns[:, :] = self.gaussian(
            energy_axis[None, :], 
            self.freqs[:, None] * constants.hbar / constants.eV * 1000, 
            collect_sigmas[:, None]
            )

        spectral_density = np.dot(self.hrf, gfxns)
        return spectral_density
