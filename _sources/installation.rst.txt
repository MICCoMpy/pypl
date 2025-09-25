.. _installation:

============
Installation
============

The recommendend installation method for **PyPL** is via Python install.
The software is tested for Python version 3.x and has the following dependencies:

   - ``numpy``
   - ``scipy``
   - ``matplotlib``
   - ``h5py``
   - ``pyyaml``
   - ``sphinx``
   - ``sphinx_rtd_theme``

To install **PyPL** and its dependencies, execute:

.. code:: bash

    $ git clone https://github.com/MICCoMpy/pypl.git
    $ cd pypl
    $ pip install .

If using **pip** is not possible, one can manually install the above dependencies, then add the directory of **PyPL** to the ``PYTHONPATH`` environment variable by, e.g.,

.. code:: bash

   # Bash shell as an example
   $ export PYTHONPATH=$PYTHONPATH:path/to/pypl

