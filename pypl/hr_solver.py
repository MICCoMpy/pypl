import numpy as np
from scipy import constants
from .utils import *
from .hr_factors import hr_factors
from .lineshape import lineshape


class hr_solver:
    """
    High-level solver for computing Huang-Rhys factors (HRFs), spectral density,
    and optical lineshapes from phonon and structural data.

    Depending on the inputs, HRFs are computed using either forces or displacements:

    - Provide only ``forces_file`` for force-based HRFs.
    - Provide both ``gs_file`` and ``es_file`` for displacement-based HRFs.

    Parameters
    ----------
    phonopy_file : str
        Path to Phonopy HDF5 file with phonon frequencies and eigenmodes.
    forces_file : str, optional
        Path to QE XML file with atomic forces.
    gs_file : str, optional
        Path to QE XML file with ground-state coordinates.
    es_file : str, optional
        Path to QE XML file with excited-state coordinates.
    mass_list : dict, optional
        Dictionary mapping element symbols to atomic masses (in atomic units).

    Raises
    ------
    ValueError
        If the input combination is invalid (e.g. mixing force and displacement inputs).
    """

    def __init__(self):
        return


    def compute_hrf_forces(self, phonon_freqs_in, phonon_modes_in, atomic_symbols, forces_in, mass_list=None):

        print("Computing Huang-Rhys factors using atomic forces.")
        print()

        freqs = phonon_freqs_in * 1e12 * 2 * np.pi
        modes = phonon_modes_in.copy()

        forces = forces_in * (constants.eV / 1e-10)

        hrf = hr_factors(freqs, modes, atomic_symbols, mass_list)
        hrf.compute_hrf_forces(forces)

        print("Total Huang-Rhys factor is % .12e" % (np.sum(hrf.hrf)))
        print()

        return {'freqs': freqs, 'hr_factors': hrf.hrf}


    def compute_hrf_dis(self, phonon_freqs_in, phonon_modes_in, atomic_symbols, gs_coord_in, es_coord_in, cell_parameters_in, mass_list=None):

        print("Computing Huang-Rhys factors using atomic displacements.")

        freqs = phonon_freqs_in * 1e12 * 2 * np.pi
        modes = phonon_modes_in.copy()

        gs_coord = gs_coord_in * 1e-10
        es_coord = es_coord_in * 1e-10
        cell_parameters = cell_parameters_in * 1e-10

        hrf = hr_factors(freqs, modes, atomic_symbols, mass_list)
        hrf.compute_hrf_dis(gs_coord, es_coord, cell_parameters)

        print("Total Huang-Rhys factor is % .12e" % (np.sum(hrf.hrf)))

        return {'freqs': freqs, 'hr_factors': hrf.hrf}



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


    def compute_spectral_density(self, hrf_dict, energy_axis=None, sigma=[6.0, 1.5]):
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
        if energy_axis is None:
            energy_axis = np.linspace(0, 200, 2001)

        nom = hrf_dict['freqs'].shape[0]

        gfxns = np.zeros((nom, energy_axis.shape[0]))

        collect_sigmas = self.f_sigma(hrf_dict['freqs'], sigma)

        gfxns[:, :] = self.gaussian(
            energy_axis[None, :], 
            hrf_dict['freqs'][:, None] * constants.hbar / constants.eV * 1000, 
            collect_sigmas[:, None]
            )

        spectral_density = np.dot(hrf_dict['hr_factors'], gfxns)
        return energy_axis, spectral_density


    def compute_lineshape_numerical_integration(self, hrf_dict, temp=4, sigma=[6, 1.5], zpl_broadening=0.3,
                                                time_range=[0, 20000], time_resolution=200001,
                                                lineshape_energy_range=[-150, 550], lineshape_energy_resolution=701):
        r"""
        Compute the temperature-dependent optical lineshape via direct time-domain integration.

        .. math::

            A(\hbar \omega, T) = \int dt \left(i \omega t - \frac{|t| \gamma_\mathrm{ZPL}}{\hbar} \right) G(t, T)

        where :math:`G(t, T)` is the generating function including phonon broadening.`\gamma_\mathrm{ZPL}` is the
        zero-phonon line (ZPL) broadening.

        Parameters
        ----------
        temp : float, optional
            Temperature in Kelvin. Default is 4.
        sigma : list of float, optional
            Linewidth interpolation values ``[sigma_min, sigma_max]`` in meV.
            Default is [6, 1.5].
        zpl_broadening : float, optional
            Additional Lorentzian broadening for the ZPL in meV. Default is 0.3.
        time_range : list of float, optional
            Time domain [start, end] in femtoseconds. Default is [0, 20000].
        time_resolution : int, optional
            Number of time points. Default is 200001.
        lineshape_energy_range : list of float, optional
            Energy range [start, end] in meV. Default is [-150, 550].
        lineshape_energy_resolution : int, optional
            Number of points in the energy axis. Default is 701.

        Notes
        -----
        The result is stored in ``self.lineshape`` as a tuple
        ``(energy_axis, A(E))`` where ``energy_axis`` is in meV and
        ``A(E)`` is the computed lineshape.
        """

        time_axis = np.linspace(time_range[0], time_range[1], time_resolution)
        lineshape_energy_axis = np.linspace(lineshape_energy_range[0], lineshape_energy_range[1], lineshape_energy_resolution)

        _lineshape = lineshape(hrf_dict)
        _lineshape.compute_lineshape_numerical_integration(temp=temp, sigma=sigma, zpl_broadening=zpl_broadening,
                                                                time_axis=time_axis, ene_axis=lineshape_energy_axis)
        self.lineshape = _lineshape.lineshape
        return self.lineshape


    def compute_lineshape_fft(self, hrf_dict, temp=4, sigma=[6, 1.5], zpl_broadening=0.3,
                              lineshape_energy_range=[-1000, 1000], lineshape_energy_resolution=2001):
        r"""
        Compute the temperature-dependent optical lineshape
        using FFT of the time-domain correlation function.

        .. math::

            A(\hbar\omega) = \frac{1}{2\pi\hbar} \!\int dt \exp \left(i \omega t - \frac{|t| \gamma_\mathrm{ZPL}}{\hbar} \right) G(t),

        where :math:`G(t, T)` is the generating function with phonon broadening.

        Parameters
        ----------
        temp : float, optional
            Temperature in Kelvin. Default is 4.
        sigma : list of float, optional
            Linewidth interpolation values ``[sigma_min, sigma_max]`` in meV.
            Default is [6, 1.5].
        zpl_broadening : float, optional
            Lorentzian broadening for the ZPL in meV. Default is 0.3.
        lineshape_energy_range : list of float, optional
            Symmetric energy range [start, end] in meV.
            Must satisfy ``start = -end``. Default is [-1000, 1000].
        lineshape_energy_resolution : int, optional
            Number of points in the energy axis. Default is 2001.

        Raises
        ------
        ValueError
            If ``lineshape_energy_range`` is not symmetric.

        Notes
        -----
        The result is stored in ``self.lineshape`` as a tuple
        ``(energy_axis, A(E))`` where ``energy_axis`` is in meV and
        ``A(E)`` is the computed lineshape.
        """

        if abs(lineshape_energy_range[0] + lineshape_energy_range[1]) > 1e-8:
            raise ValueError("Energy range must be symmetric around zero: expected start = -end.")

        energy_axis = np.linspace(lineshape_energy_range[0], lineshape_energy_range[1], lineshape_energy_resolution)

        _lineshape = lineshape(hrf_dict)
        _lineshape.compute_lineshape_fft(temp=temp, sigma=sigma, zpl_broadening=zpl_broadening, energy_axis=energy_axis)
        self.lineshape = _lineshape.lineshape
        return self.lineshape


    def compute_spectrum(self, ezpl=1.0, spectrum_type='PL', lineshape=None):
        """
        Compute the normalized photoluminescence (PL) or absorption spectrum.

        Parameters
        ----------
        spectrum_type : str
            Spectrum type:
            - "PL": photoluminescence spectrum
            - "Abs": absorption spectrum
        ezpl : float
            Zero-phonon line (ZPL) energy in meV.

        Returns
        -------
        energy_axis : ndarray
            Energy axis in meV.
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
