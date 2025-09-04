import numpy as np
from scipy import constants
from .utils import *
from .hrf import hrf
from .lineshape import lineshape


class hr_solver:
    """
    High-level solver for computing Huang–Rhys factors (HRFs), spectral density,
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

    def __init__(self, phonopy_file, forces_file=None, gs_file=None, es_file=None, mass_list=None):

        freqs, modes = parse_phonopy_h5(phonopy_file)

        self.freqs = freqs * 1e12 * 2 * np.pi
        self.modes = modes

        if forces_file is not None and gs_file is None and es_file is None:
            print("Computing Huang-Rhys factors using atomic forces.")

            forces = parse_forces_qexml(forces_file)
            forces *= (constants.eV / 1e-10)
            atomic_symbols, _, _ = parse_atoms_qexml(forces_file)

            self.hrf = hrf(self.freqs, self.modes, atomic_symbols)
            self.hrf.set_masses(mass_list)
            self.hrf.compute_hrf_forces(forces)

        elif forces_file is None and gs_file is not None and es_file is not None:
            print("Computing Huang-Rhys factors using atomic displacements.")

            atomic_symbols, gs_coord, cell_parameters = parse_atoms_qexml(gs_file)
            atomic_symbols_2, es_coord, cell_parameters_2 = parse_atoms_qexml(es_file)

            assert atomic_symbols == atomic_symbols_2, (
                "Mismatch in atomic symbols between ground-state and excited-state structures. "
                "Please ensure both structures have the same atom types and order."
            )

            assert np.max(np.abs(cell_parameters - cell_parameters_2)) < 1e-12, (
                "Mismatch in cell parameters between ground-state and excited-state structures. "
                "Ensure both structures are in the same unit cell."
            )

            gs_coord *= 1e-10
            es_coord *= 1e-10
            cell_parameters *= 1e-10

            self.hrf = hrf(self.freqs, self.modes, atomic_symbols)
            self.hrf.set_masses(mass_list)
            self.hrf.compute_hrf_dis(gs_coord, es_coord, cell_parameters)

        else:
            raise ValueError(
                "Invalid input combination:\n"
                "- To use forces, provide `forces_file` only.\n"
                "- To use displacements, provide both `gs_file` and `es_file`.\n"
                "- Do not mix force and displacement inputs."
            )

        self.tot_hrf = sum(self.hrf.hrf)


    def compute_spectral_density(self, spectral_density_energy_axis=None, sigma=[6, 1.5]):
        r"""
        Compute the phonon spectral density.

        .. math::

            S(\hbar \omega) = \sum_k S_k G(\hbar \omega, \hbar \omega_k, \sigma(\omega_k))

        where :math:`G` is a normalized Gaussian centered at each mode.

        Parameters
        ----------
        spectral_density_energy_axis : ndarray, optional
            Energy axis (in meV). Defaults to ``np.linspace(0, 200, 2001)``
            (0–200 meV in 0.1 meV steps).
        sigma : list of float, optional
            Two-element list ``[sigma_min, sigma_max]`` in meV for linewidth
            interpolation. Default is [6, 1.5].

        Returns
        -------
        energy_axis : ndarray
            Energy axis in meV.
        spectral_density : ndarray
            Computed spectral density (in 1/meV).
        """

        if spectral_density_energy_axis is None:
            spectral_density_energy_axis = np.linspace(0, 200, 2001)

        spectral_density = self.hrf.compute_spectral_density(spectral_density_energy_axis, sigma)
        return spectral_density_energy_axis, spectral_density


    def compute_lineshape_numerical_integration(self, temp=4, sigma=[6, 1.5], zpl_broadening=0.3,
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

        self._lineshape = lineshape(self.hrf)
        self._lineshape.compute_lineshape_numerical_integration(temp=temp, sigma=sigma, zpl_broadening=zpl_broadening,
                                                                time_axis=time_axis, ene_axis=lineshape_energy_axis)
        self.lineshape = self._lineshape.lineshape


    def compute_lineshape_fft(self, temp=4, sigma=[6, 1.5], zpl_broadening=0.3,
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

        self._lineshape = lineshape(self.hrf)
        self._lineshape.compute_lineshape_fft(temp=temp, sigma=sigma, zpl_broadening=zpl_broadening, energy_axis=energy_axis)
        self.lineshape = self._lineshape.lineshape


    def compute_spectrum(self, spectrum_type, ezpl):
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

        # multiply the pre-coeffcient
        if spectrum_type == "PL":
            energy_axis = ezpl - self._lineshape.lineshape[0]
            spectrum = self._lineshape.lineshape[1] * energy_axis**3
        elif spectrum_type == "Abs":
            energy_axis = ezpl + self._lineshape.lineshape[0]
            spectrum = self._lineshape.lineshape[1] * energy_axis

        # normalization
        spectrum = spectrum / sum(spectrum) / abs(energy_axis[1] - energy_axis[0]) * 1000

        return energy_axis, spectrum
