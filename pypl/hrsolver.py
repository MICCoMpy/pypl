import numpy as np
from scipy import constants
from .utils import *
from .hrf import hrf
from .lineshape import lineshape


class HRsolver:

    def __init__(self, phonopy_file, forces_file=None, gs_file=None, es_file=None, mass_list=None):
        """
        Initialize the HRF object and determine whether to use forces or displacements to compute Huang-Rhys factors.

        Parameters
        ----------
        phonopy_file : str
            Path to the Phonopy HDF5 file containing phonon frequencies and modes.
        forces_file : str, optional
            Path to the QE xml file containing forces (used for force-based HRF computation).
        gs_file : str, optional
            Path to QE xml file for the ground-state geometry (used for displacement-based HRF).
        es_file : str, optional
            Path to the QE xml file for the excited-state geometry (used for displacement-based HRF).
        mass_list : dict, optional
            Dictionary mapping element symbols to atomic masses (in atomic units).
        """

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


    def compute_spectral_density(self, spectral_density_energy_axis=np.linspace(0, 200, 2001), sigma=[6, 1.5]):
        """
        Compute the phonon spectral density.

        This function evaluates:
            S(hbar * omega) = sum_k [ S_k * G(hbar * omega, hbar * omega_k, sigma_k) ]

        Parameters
        ----------
        spectral_density_energy_axis : ndarray, optional
            Energy axis (in meV) over which to evaluate the spectral density.
            Defaults to np.linspace(0, 200, 2001), i.e., 0 to 200 meV in 0.1 meV steps.
        sigma : list of float, optional
            Linewidth parameters [sigma_min, sigma_max] for frequency-dependent broadening (in meV).
            Defaults to [6, 1.5]

        Returns
        -------
        energy_axis : ndarray
            Energy axis used for the spectral density (in meV).
        spectral_density : ndarray
            Computed spectral density at each energy point (in 1 / meV).
        """

        spectral_density = self.hrf.compute_spectral_density(spectral_density_energy_axis, sigma)
        return spectral_density_energy_axis, spectral_density


    def compute_lineshape_numerical_integration(self, temp=4, sigma=[6, 1.5], zpl_broadening=0.3,
                                                time_range=[0, 20000], time_resolution=200001,
                                                lineshape_energy_range=[-150, 550], lineshape_energy_resolution=701):
        """
        Compute the temperature-dependent line shape function A(hbar * omega, T).

        Parameters
        ----------
        temp : float, optional
            Temperature in Kelvin. Default is 4 K.
        sigma : list of float, optional
            Linewidth interpolation values [sigma_min, sigma_max] for Gaussian broadening (in meV). Default is [6, 1.5]
        zpl_broadening : float, optional
            Additional Lorentzian broadening for the zero-phonon line (in meV). Default is 0.3 meV.
        time_range : list of float, optional
            Time domain range [start, end] in femtoseconds. Default is [0, 20000] fs.
        time_resolution : int, optional
            Number of time points in the time axis. Default is 200001.
        lineshape_energy_range : list of float, optional
            Energy range [start, end] in meV for computing the lineshape. Default is [-150, 550] meV.
        lineshape_energy_resolution : int, optional
            Number of points in the energy axis. Default is 701.

        Returns
        -------
        None
            The result is stored in self.lineshape as a tuple (energy_axis, A(omega, T)).
        """

        time_axis = np.linspace(time_range[0], time_range[1], time_resolution)
        lineshape_energy_axis = np.linspace(lineshape_energy_range[0], lineshape_energy_range[1], lineshape_energy_resolution)

        self._lineshape = lineshape(self.hrf)
        self._lineshape.compute_lineshape_numerical_integration(temp=temp, sigma=sigma, zpl_broadening=zpl_broadening,
                                                                time_axis=time_axis, ene_axis=lineshape_energy_axis)
        self.lineshape = self._lineshape.lineshape


    def compute_lineshape_fft(self, temp=4, sigma=[6, 1.5], zpl_broadening=0.3,
                              lineshape_energy_range=[-1000, 1000], lineshape_energy_resolution=2001):
        """
        Compute the temperature-dependent line shape function A(hbar * omega, T).

        Parameters
        ----------
        temp : float, optional
            Temperature in Kelvin. Default is 4 K.
        sigma : list of float, optional
            Linewidth interpolation values [sigma_min, sigma_max] for Gaussian broadening (in meV). Default is [6, 1.5]
        zpl_broadening : float, optional
            Additional Lorentzian broadening for the zero-phonon line (in meV). Default is 0.3 meV.
        lineshape_energy_range : list of float, optional
            Energy range [start, end] in meV for computing the lineshape. Default is [-1000, 1000] meV.
        lineshape_energy_resolution : int, optional
            Number of points in the energy axis. Default is 2001.

        Returns
        -------
        None
            The result is stored in self.lineshape as a tuple (energy_axis, A(omega, T)).
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
            Type of spectrum to compute. Options:
            - "PL" : photoluminescence spectrum
            - "Abs": absorption spectrum
        ezpl : float
            Zero-phonon line (ZPL) energy in meV.

        Returns
        -------
        energy_axis : ndarray
            Energy axis (in meV) of the computed spectrum.
        spectrum : ndarray
            Normalized spectral intensity over the energy axis.
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
