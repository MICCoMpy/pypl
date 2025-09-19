import numpy as np
from scipy import constants
from .utils import *
from .hr_factors import hr_factors
from .lineshape import lineshape


class hr_solver:
    """
    High-level solver for computing Huang-Rhys factors (HRFs), spectral density,
    and optical lineshapes from phonon and structural data.
    """

    def __init__(self):
        return

    def compute_hrf_forces(self, phonon_freqs_in, phonon_modes_in, atomic_symbols, forces_in, mass_list=None):
        """
        Compute Huang-Rhys factors using atomic forces.

        Parameters
        ----------
        phonon_freqs_in : ndarray of shape (M,)
            Phonon frequencies in THz.
        phonon_modes_in : ndarray of shape (M, 3N)
            Phonon eigenmodes in Cartesian representation.
        atomic_symbols : list of str
            Atomic symbols for all atoms.
        forces_in : ndarray of shape (N, 3)
            Atomic forces in eV/Å.
        mass_list : dict, optional
            Dictionary mapping element symbols to atomic masses (in a.u.).

        Returns
        -------
        dict
            Dictionary with keys:
            - ``'freqs'`` : ndarray of shape (M,), phonon frequencies in rad/s.
            - ``'hr_factors'`` : ndarray of shape (M,), Huang–Rhys factors :math:`S_k`.
        """

        print("Computing Huang-Rhys factors using atomic forces.")
        print()

        freqs = phonon_freqs_in * 1e12 * 2 * np.pi
        modes = phonon_modes_in.copy()

        forces = forces_in * (constants.eV / 1e-10)

        hrf = hr_factors(freqs, modes, atomic_symbols, mass_list)
        hrf.compute_hrf_forces(forces)

        print("Total Huang-Rhys factor is % .12e" % (np.sum(hrf.hrf)))
        print()

        return {"freqs": freqs, "hr_factors": hrf.hrf}

    def compute_hrf_dis(
        self,
        phonon_freqs_in,
        phonon_modes_in,
        atomic_symbols,
        gs_coord_in,
        es_coord_in,
        cell_parameters_in,
        mass_list=None,
    ):
        """
        Compute Huang-Rhys factors using atomic displacements between
        ground-state (GS) and excited-state (ES) geometries.

        Parameters
        ----------
        phonon_freqs_in : ndarray of shape (M,)
            Phonon frequencies in THz.
        phonon_modes_in : ndarray of shape (M, 3N)
            Phonon eigenmodes in Cartesian representation.
        atomic_symbols : list of str
            Atomic symbols for all atoms.
        gs_coord_in : ndarray of shape (N, 3)
            Ground-state atomic coordinates in Å.
        es_coord_in : ndarray of shape (N, 3)
            Excited-state atomic coordinates in Å.
        cell_parameters_in : ndarray of shape (3, 3)
            Lattice vectors in Å.
        mass_list : dict, optional
            Dictionary mapping element symbols to atomic masses (in a.u.).

        Returns
        -------
        dict
            Dictionary with keys:
            - ``'freqs'`` : ndarray of shape (M,), phonon frequencies in rad/s.
            - ``'hr_factors'`` : ndarray of shape (M,), Huang–Rhys factors :math:`S_k`.
        """

        print("Computing Huang-Rhys factors using atomic displacements.")

        freqs = phonon_freqs_in * 1e12 * 2 * np.pi
        modes = phonon_modes_in.copy()

        gs_coord = gs_coord_in * 1e-10
        es_coord = es_coord_in * 1e-10
        cell_parameters = cell_parameters_in * 1e-10

        hrf = hr_factors(freqs, modes, atomic_symbols, mass_list)
        hrf.compute_hrf_dis(gs_coord, es_coord, cell_parameters)

        print("Total Huang-Rhys factor is % .12e" % (np.sum(hrf.hrf)))

        return {"freqs": freqs, "hr_factors": hrf.hrf}

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
        exponent = np.exp(-((x - mu) ** 2) / (2 * sigma**2))
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

    def compute_spectral_density(self, hrf_dict, energy_axis=None, sigma=[6.0, 1.5]):
        r"""
        Compute the phonon spectral density with Gaussian broadening.

        .. math::

            S(\hbar\omega) =
            \sum_k S_k \, G(\hbar\omega; \hbar\omega_k, \sigma(\omega_k))

        where :math:`S_k` is the partial Huang–Rhys factor and
        :math:`G` is a normalized Gaussian.

        Parameters
        ----------
        hrf_dict : dict
            Dictionary containing:
            - ``'freqs'`` : ndarray of phonon frequencies (rad/s).
            - ``'hr_factors'`` : ndarray of partial Huang–Rhys factors.
        energy_axis : ndarray of shape (N,), optional
            Energy axis in meV over which to evaluate the spectral density.
            If None, defaults to ``np.linspace(0, 200, 2001)``.
        sigma : list of float, optional
            Linewidth interpolation values ``[sigma_0, sigma_1]`` in meV.

        Returns
        -------
        energy_axis : ndarray of shape (N,)
            Energy axis in meV.
        spectral_density : ndarray of shape (N,)
            Spectral density :math:`S(\hbar\omega)` in 1/meV.
        """

        if energy_axis is None:
            energy_axis = np.linspace(0, 200, 2001)

        nom = hrf_dict["freqs"].shape[0]

        gfxns = np.zeros((nom, energy_axis.shape[0]))

        collect_sigmas = self.f_sigma(hrf_dict["freqs"], sigma)

        gfxns[:, :] = self.gaussian(
            energy_axis[None, :],
            hrf_dict["freqs"][:, None] * constants.hbar / constants.eV * 1000,
            collect_sigmas[:, None],
        )

        spectral_density = np.dot(hrf_dict["hr_factors"], gfxns)
        return energy_axis, spectral_density

    def compute_lineshape_numerical_integration(
        self,
        hrf_dict,
        temp=4,
        sigma=[6, 1.5],
        zpl_broadening=0.3,
        time_range=[0, 20000],
        time_resolution=200001,
        lineshape_energy_range=[-150, 550],
        lineshape_energy_resolution=701,
    ):
        r"""
        Compute the temperature-dependent optical lineshape
        via direct numerical integration in the time domain.

        .. math::

            A(\hbar\omega, T) =
            \int dt \,
            \exp\!\left(i \omega t - \frac{|t|\gamma_\mathrm{ZPL}}{\hbar}\right) G(t, T)

        where :math:`G(t, T)` is the generating function with phonon broadening,
        and :math:`\gamma_\mathrm{ZPL}` is the zero-phonon line (ZPL) broadening.

        Parameters
        ----------
        hrf_dict : dict
            Dictionary containing phonon frequencies and HR factors.
        temp : float, optional
            Temperature in Kelvin. Default is 4.
        sigma : list of float, optional
            Linewidth interpolation values ``[sigma_min, sigma_max]`` in meV.
        zpl_broadening : float, optional
            Additional Lorentzian broadening of the ZPL in meV. Default is 0.3.
        time_range : list of float, optional
            Time domain ``[start, end]`` in fs. Default is [0, 20000].
        time_resolution : int, optional
            Number of time points. Default is 200001.
        lineshape_energy_range : list of float, optional
            Energy range ``[start, end]`` in meV. Default is [-150, 550].
        lineshape_energy_resolution : int, optional
            Number of energy points. Default is 701.

        Returns
        -------
        tuple
            ``(energy_axis, A(E))``, where:
            - ``energy_axis`` : ndarray of energies in meV.
            - ``A(E)`` : ndarray of the computed lineshape.
        """

        time_axis = np.linspace(time_range[0], time_range[1], time_resolution)
        lineshape_energy_axis = np.linspace(
            lineshape_energy_range[0], lineshape_energy_range[1], lineshape_energy_resolution
        )

        _lineshape = lineshape(hrf_dict)
        _lineshape.compute_lineshape_numerical_integration(
            temp=temp, sigma=sigma, zpl_broadening=zpl_broadening, time_axis=time_axis, ene_axis=lineshape_energy_axis
        )
        self.lineshape = _lineshape.lineshape
        return self.lineshape

    def compute_lineshape_fft(
        self,
        hrf_dict,
        temp=4,
        sigma=[6, 1.5],
        zpl_broadening=0.3,
        lineshape_energy_range=[-1000, 1000],
        lineshape_energy_resolution=2001,
    ):
        r"""
        Compute the temperature-dependent optical lineshape
        using FFT of the time-domain correlation function.

        .. math::

            A(\hbar\omega, T) =
            \frac{1}{2\pi\hbar} \int dt \,
            \exp\!\left(i \omega t - \frac{|t|\gamma_\mathrm{ZPL}}{\hbar}\right) G(t, T)

        Parameters
        ----------
        hrf_dict : dict
            Dictionary containing phonon frequencies and HR factors.
        temp : float, optional
            Temperature in Kelvin. Default is 4.
        sigma : list of float, optional
            Linewidth interpolation values ``[sigma_min, sigma_max]`` in meV.
        zpl_broadening : float, optional
            Lorentzian broadening of the ZPL in meV. Default is 0.3.
        lineshape_energy_range : list of float, optional
            Symmetric energy range ``[start, end]`` in meV (must satisfy start = -end).
            Default is [-1000, 1000].
        lineshape_energy_resolution : int, optional
            Number of energy points. Default is 2001.

        Returns
        -------
        tuple
            ``(energy_axis, A(E))``, where:
            - ``energy_axis`` : ndarray of energies in meV.
            - ``A(E)`` : ndarray of the computed lineshape.

        Raises
        ------
        ValueError
            If ``lineshape_energy_range`` is not symmetric about zero.
        """

        if abs(lineshape_energy_range[0] + lineshape_energy_range[1]) > 1e-8:
            raise ValueError("Energy range must be symmetric around zero: expected start = -end.")

        energy_axis = np.linspace(lineshape_energy_range[0], lineshape_energy_range[1], lineshape_energy_resolution)

        _lineshape = lineshape(hrf_dict)
        _lineshape.compute_lineshape_fft(temp=temp, sigma=sigma, zpl_broadening=zpl_broadening, energy_axis=energy_axis)
        self.lineshape = _lineshape.lineshape
        return self.lineshape

    def compute_spectrum(self, ezpl=1.0, spectrum_type="PL", lineshape=None):
        """
        Compute the normalized photoluminescence (PL) or absorption spectrum
        from a precomputed lineshape.

        Parameters
        ----------
        ezpl : float, optional
            Zero-phonon line (ZPL) energy in meV. Default is 1.0.
        spectrum_type : {'PL', 'Abs'}, optional
            Type of spectrum to compute:
            - ``'PL'`` : photoluminescence spectrum
            - ``'Abs'`` : absorption spectrum
        lineshape : tuple, optional
            Precomputed lineshape ``(energy_axis, A(E))``.
            If None, uses ``self.lineshape``.

        Returns
        -------
        energy_axis_out : ndarray
            Energy axis in meV, shifted relative to the ZPL.
        spectrum : ndarray
            Normalized spectral intensity.
        """

        if lineshape is None:
            lineshape = self.lineshape

        # multiply the pre-coeffcient
        if spectrum_type == "PL":
            energy_axis_out = ezpl - lineshape[0]
            spectrum = lineshape[1] * energy_axis_out**3
        elif spectrum_type == "Abs":
            energy_axis_out = ezpl + lineshape[0]
            spectrum = lineshape[1] * energy_axis_out
        else:
            raise ValueError("Invalid spectrum type")

        # normalization
        spectrum = spectrum / sum(spectrum) / abs(energy_axis_out[1] - energy_axis_out[0]) * 1000

        return energy_axis_out, spectrum
