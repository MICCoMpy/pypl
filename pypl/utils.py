import numpy as np
import xml.etree.ElementTree as ET
import h5py
import yaml


ry2ev = 13.605693122994017
bohr2ang = 0.529177210903


def parse_atoms_qexml(fileName):
    """
    Parse atomic structure information from a Quantum ESPRESSO QE XML output file.

    Parameters
    ----------
    fileName : str
        Path to the QE XML file.

    Returns
    -------
    atomic_symbols : list of str
        List of atomic symbols for all atoms in the structure.
    atomic_positions : numpy.ndarray of shape (N, 3)
        Cartesian atomic coordinates in Ångström.
    cell_parameters : numpy.ndarray of shape (3, 3)
        Lattice vectors (cell parameters) in Ångström.
    """

    tree = ET.parse(fileName)
    root = tree.getroot()

    atomic_structure = root.find("output/atomic_structure")

    atomic_symbols = []
    atomic_positions = []

    for atom in atomic_structure.find("atomic_positions").findall("atom"):
        name = atom.attrib["name"]
        coords = list(map(float, atom.text.strip().split()))
        atomic_symbols.append(name)
        atomic_positions.append(coords)
    atomic_positions = np.array(atomic_positions, dtype=np.float64) * bohr2ang

    cell_parameters = []
    for i in [1, 2, 3]:
        a = atomic_structure.find(f"cell/a{i}").text.split()
        cell_parameters.append(a)
    cell_parameters = np.array(cell_parameters, dtype=np.float64) * bohr2ang

    return atomic_symbols, atomic_positions, cell_parameters


def parse_forces_qexml(fileName):
    """
    Parse atomic forces from a Quantum ESPRESSO QE XML output file.

    Parameters
    ----------
    fileName : str
        Path to the QE XML file.

    Returns
    -------
    forces : numpy.ndarray of shape (N, 3)
        Forces on atoms in eV / Ångström.
    """

    tree = ET.parse(fileName)
    root = tree.getroot()

    forces_tag = root.find("output/forces")
    if forces_tag is None:
        raise ValueError("Could not find <forces> tag in <output>.")

    force_values = np.fromstring(forces_tag.text, sep=' ', dtype=np.float64)
    forces = force_values.reshape((-1, 3)) * 2.0 * ry2ev / bohr2ang

    return forces


def parse_phonopy_h5(fileName, real_tol=1e-10):
    """
    Parse phonon frequencies and eigenmodes from a Phonopy HDF5 file.

    Parameters
    ----------
    fileName : str
        Path to the Phonopy HDF5 file (e.g., phonon.hdf5).
    real_tol : float, optional
        Tolerance below which imaginary components of eigenmodes are discarded.
        Default is 1e-10.

    Returns
    -------
    freqs : ndarray of shape (M,)
        Phonon frequencies in THz.
    modes : ndarray of shape (M, Nat, 3)
        Phonon eigenmodes. Converted to real if imaginary parts are negligible.
    """

    with h5py.File(fileName, "r") as f:
        freqs = np.array(list(f.values())[1][0], dtype=np.float64)
        modes = np.array(list(f.values())[0][0])

    modes = np.swapaxes(modes, 0, 1)
    modes = modes.reshape(freqs.shape[0], -1, 3)

    max_imag = np.max(np.abs(modes.imag))
    if max_imag > real_tol:
        raise ValueError(f"Imaginary part of eigenmodes exceeds tolerance: max |Im| = {max_imag:.2e}")

    modes = modes.real
    return freqs, modes


def parse_phonopy_yaml(fileName="band.yaml", real_tol=1e-10):
    """
    Parse phonon frequencies and eigenmodes from a Phonopy band.yaml file.

    Parameters
    ----------
    filename : str, optional
        Path to the band.yaml file (default is "band.yaml").

    Returns
    -------
    frequencies : ndarray of shape (3N,)
        Phonon frequencies in THz.
    modes : ndarray of shape (3N, N, 3)
        Phonon eigenmodes.
    """

    with open(fileName, 'r') as f:
        data = yaml.safe_load(f)

    bands = data["phonon"][0]["band"]
    num_modes = len(bands)
    num_atoms = len(bands[0]["eigenvector"])

    freqs = np.zeros(num_modes)
    modes = np.zeros((num_modes, num_atoms, 3))

    for i, band in enumerate(bands):
        # Frequency in THz
        freq_thz = band["frequency"]
        freqs[i] = float(freq_thz)

        for a, atom_vec in enumerate(band["eigenvector"]):
            for d in range(3):
                real, imag = atom_vec[d]
                if abs(imag) > real_tol:
                    raise ValueError(
                        f"Imaginary part too large at mode {i}, atom {a}, direction {d}: "
                        f"|Im| = {imag:.2e} > {real_tol:.1e}"
                    )
                modes[i, a, d] = float(real)

    return freqs, modes
