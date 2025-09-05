.. _installation:

============
Installation
============

The recommendend installation method for **PyPL** is via python install. 
The software is tested for python version 3.x and has the following dependencies: 

   - ``numpy``
   - ``scipy``
   - ``matplotlib``
   - ``h5py``
   - ``pyyaml``
   

The dependencies will all be installed automatically, following instructions reported below.  


Source Code Installation
========================

To install **PyPL** you need to execute:  

.. code:: bash

    $ git clone https://github.com/jinyuchem/pypl.git
    $ cd pypl
    $ pip install -e .

If using **pip** is not possible, one can manually install the above dependencies, and then add the directory of **PyPL** to the ``PYTHONPATH`` environment variable by, e.g.,

.. code:: bash

   # Bash shell as an example
   $ export PYTHONPATH=$PYTHONPATH:path/to/pypl

