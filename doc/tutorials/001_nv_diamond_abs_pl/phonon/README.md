Steps of phonon calculations using Phonopy and Quantum ESPRESSO

1. Generate displacements (from the ground state structure)
```
    phonopy --qe -d --dim="1 1 1" -c gs-dft-pw.in -v
```

2. Perform DFT calculations using the displaced structures

3. Collect forces
```
    phonopy --qe -f supercell-{001..574}.out
```

4. Compute phonon frequencies and modes
```
    phonopy --dim="1 1 1" --fc-symmetry --mesh="1 1 1" --eigenvectors --writefc --qe -c gs-dft-pw.in --mesh-format=hdf5
```
