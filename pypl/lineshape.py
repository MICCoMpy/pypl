import numpy as np
from scipy.integrate import simpson as simps
from scipy import constants
import warnings


class lineshape:

    def __init__(self, hrf):
        """Initialize the lineshape class.

        Args:
            hrf (.hrf.hrf): hrf class.
        """

        self.hrf = hrf


    def f_sigma(self):
        """
        Calculate the broadening parameter :math:`\sigma` for all phonons.
        """

        self.all_sigma = self.sigma[0] - (self.sigma[0] - self.sigma[1]) / (max(self.hrf.freqs) - min(self.hrf.freqs)) * (self.hrf.freqs[:] - min(self.hrf.freqs))


    def generate_fxn(self):
        """Compute the generating function :math:`G(t, T)`.

        Returns:
            (tuple): tuple containing:
                real_gfxn (np.array): real part of :math:`G(t, T)`.
                imag_gfxn (np.array): imag part of :math:`G(t, T)`.
        """

        ReS_t = (np.exp(-self.time_axis[None, :]**2 * self.all_sigma[:, None]**2 / 2 /constants.hbar**2)
                 * np.cos(self.hrf.freqs[:, None] * self.time_axis[None, :]))
        ImS_t = (np.exp(-self.time_axis[None, :]**2 * self.all_sigma[:, None]**2 / 2 / constants.hbar**2) 
                 * (-1) * np.sin(self.hrf.freqs[:, None] * self.time_axis[None, :]))
        ReS_t = np.dot(self.hrf.hrf, ReS_t)
        ImS_t = np.dot(self.hrf.hrf, ImS_t)

        ReS_0 = np.sum(self.hrf.hrf)

        if self.temp > 0:
            ReC_t = (np.exp(-self.time_axis[None, :]**2 * self.all_sigma[:, None]**2 / 2 / constants.hbar**2)
                      * np.cos(self.hrf.freqs[:, None] * self.time_axis[None, :]) * self.ph_occ[:, None])
            ReC_t = np.dot(self.hrf.hrf, ReC_t)

            ReC_0 = np.sum(self.hrf.hrf * self.ph_occ)

            real_gfxn = np.exp(ReS_t - ReS_0 + 2 * ReC_t - 2 * ReC_0) * np.cos(ImS_t)
            imag_gfxn = np.exp(ReS_t - ReS_0 + 2 * ReC_t - 2 * ReC_0) * np.sin(ImS_t)

        elif abs(self.temp) < 1e-8:

            real_gfxn = np.exp(ReS_t - ReS_0) * np.cos(ImS_t)
            imag_gfxn = np.exp(ReS_t - ReS_0) * np.sin(ImS_t)

        return real_gfxn, imag_gfxn


    def compute_lineshape_numerical_integration(self, temp, sigma, zpl_broadening, time_axis, ene_axis):
        """
        Compute the lineshape function :math:`A(\hbar\omega, T)`.

        Args:
            temp (float): temperature (K).
            sigma (list<float>): broadening parameter for Huang-Rhys factors (meV).
            lamba (float): broadening parameter for the zero-phonon line (meV).
            time_axis (np.array): time axis (fs).
            ene_axis (np.array): energy axis (meV).
        """
        self.temp = temp
        self.sigma = np.array(sigma) * constants.eV * 1e-3
        self.zpl_broadening = zpl_broadening * constants.eV * 1e-3
        self.time_axis = time_axis * 1e-15
        self.ene_axis = ene_axis * constants.eV * 1e-3
        self.f_sigma()
        self.ph_occ = 1 / (np.exp((self.hrf.freqs * constants.hbar) / (constants.Boltzmann * self.temp)) - 1)

        gr, gi = self.generate_fxn()

        ex = np.exp( - (np.abs(self.time_axis) * self.zpl_broadening / constants.hbar))

        theta = self.time_axis[None, :] * self.ene_axis[:, None] / constants.hbar
        integrand = (gr[None, :] * np.cos(theta) - gi[None, :] * np.sin(theta)) * ex[None, :]
        lineshape = simps(integrand, x=self.time_axis, axis=1)

        # unit conversion for lineshape
        lineshape = lineshape / constants.hbar / np.pi * constants.eV
 
        # unit from Joule to meV
        new_ene_axis = self.ene_axis / constants.eV * 1000
        self.lineshape = (new_ene_axis, lineshape)


    def compute_lineshape_fft(self, temp, sigma, zpl_broadening, energy_axis):
        """
        Compute the lineshape function A(E, T) via time-domain correlation and FFT.

        This evaluates the time-domain correlation function:

            G(t) = exp[ S(t) - S(0) + 2*C(t) - 2*C(0) ] * exp( -|t| * gamma_zpl / hbar )

        where:
            - S(t)  = sum_k S_k * exp( - (sigma_k/hbar)^2 * t^2 / 2 ) * exp( i * omega_k * t )
            - C(t)  = sum_k S_k * n_k * exp( - (sigma_k/hbar)^2 * t^2 / 2 ) * cos( omega_k * t )
            - gamma_zpl = ZPL Lorentzian broadening in meV

        Args:
            temp (float): temperature (K)
            sigma (list of float): phonon broadening parameters in meV, e.g. [6.0, 1.5]
            zpl_broadening (float): zero-phonon line broadening (HWHM) in meV
            energy_axis (np.ndarray): target energy axis for FFT (in meV)

        Sets:
            self.lineshape: tuple (energy_axis_meV, A(E)), where A(E) is the lineshape.
        """

        self.temp = temp
        # meV to J
        self.sigma = np.array(sigma) * constants.eV * 1e-3
        self.zpl_broadening = zpl_broadening * constants.eV * 1e-3
        self.energy_axis = energy_axis * constants.eV * 1e-3

        # Construct time axis from energy axis using FFT frequency convention
        d_energy = self.energy_axis[1] - self.energy_axis[0]  # Energy spacing in J
        d_omega = d_energy / constants.hbar  # Angular frequency spacing in rad/s
        
        n_points = len(self.energy_axis)
        d_t = 2 * np.pi / (n_points * d_omega)  # Time spacing
        t_max = n_points * d_t / 2
        time_axis = np.linspace(-t_max, t_max, n_points, endpoint=True)

        print('time_axis range:', time_axis.min(), 'to', time_axis.max())
        print('d_t (s):', d_t)

        self.f_sigma()
        self.ph_occ = 1 / (np.exp((self.hrf.freqs * constants.hbar) / (constants.Boltzmann * self.temp)) - 1)

        # Compute S(t)
        gauss_t = np.exp(-0.5 * (self.all_sigma[:, None]**2 / constants.hbar**2) * (time_axis[None, :]**2))
        phase_t = np.exp(1j * self.hrf.freqs[:, None] * time_axis[None, :])
        S_t_modes = gauss_t * phase_t
        S_t = np.dot(self.hrf.hrf, S_t_modes)
        S_0 = np.sum(self.hrf.hrf)

        # Compute C(t) if T > 0
        if temp > 0.0:
            cos_t = np.cos(self.hrf.freqs[:, None] * time_axis[None, :])
            C_t_modes = gauss_t * cos_t * self.ph_occ[:, None]
            C_t = np.dot(self.hrf.hrf, C_t_modes)
            C_0 = np.sum(self.hrf.hrf * self.ph_occ)
            exponent = (S_t - S_0) + 2.0 * (C_t - C_0)
        else:
            exponent = (S_t - S_0)

        # Time-domain ZPL damping
        zpl_damping = np.exp(-np.abs(time_axis) * (self.zpl_broadening / constants.hbar))

        # Final G(t)
        G_t = np.exp(exponent) * zpl_damping

        # FFT
        G_t_shifted = np.fft.ifftshift(G_t)
        F_w = d_t * np.fft.fftshift(np.fft.fft(G_t_shifted))

        # Construct the corresponding angular frequency axis
        omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n_points, d_t))  # rad/s

        # Convert to energy axis in meV
        energy_axis_out = (constants.hbar * omega) / (constants.eV * 1e-3)  # meV

        print('Energy range (meV):', energy_axis_out.min(), 'to', energy_axis_out.max())
        print('d_E (meV):', energy_axis_out[1] - energy_axis_out[0])

        # Check imaginary part
        imag_ratio = np.linalg.norm(F_w.imag) / np.linalg.norm(F_w.real)
        if imag_ratio > 1e-10:
            warnings.warn(
                f"Non-negligible imaginary component in FFT result: |Im(F_w)| / |Re(F_w)| = {imag_ratio:.2e}. "
                "This may indicate numerical noise or asymmetry in G(t). Using only the real part.",
                UserWarning
            )

        lineshape = np.real(F_w)

        # Sort output by energy (should already be sorted, but just to be safe)
        sort_idx = np.argsort(energy_axis_out)
        energy_axis_out = energy_axis_out[sort_idx]
        lineshape = lineshape[sort_idx]

        # Normalize
        lineshape = lineshape / constants.hbar * constants.eV / 2 / np.pi

        print('Integral check:', np.sum(lineshape) * (energy_axis_out[1] - energy_axis_out[0]))

        self.lineshape = (energy_axis_out, lineshape)
