#!/usr/bin/env python3

import numpy as np
import sys
import os
import shutil

# coordinates along the c.c.: 0.0 is the gs structure; 1.0 is the es structure
COD = np.linspace(-0.2, 1.2, 15)
# file contains the gs structure
GS_file = "gs_coord.in"
# file contains the es structure
ES_file = "es_cdft_010_coord.in"
# file contains other setting for pw
Pre_files = "prefix.in"
# file contains the resulting input file for pw
Final_files = "pw.in"


GS = np.loadtxt(GS_file, usecols=(1, 2, 3))
ES = np.loadtxt(ES_file, usecols=(1, 2, 3))

DIFF = ES - GS

ATOMS = np.genfromtxt(GS_file, dtype="str", usecols=0)


path_old = os.getcwd() + "/"

for i in range(len(COD)):
    path_new = path_old + "Image-" + str(i + 1) + "/"
    os.mkdir(path_new)
    filename_o = path_old + Pre_files
    filename_n = path_new + Final_files
    shutil.copy2(filename_o, filename_n)
    os.chdir(path_new)

    CO_n = GS + DIFF * COD[i]

    with open(Final_files, "a") as fn:
        for j in range(GS.shape[0]):
            nline = (
                str(ATOMS[j]),
                "   % .12f   % .12f   % .12f" % (CO_n[j, 0], CO_n[j, 1], CO_n[j, 2]),
                "\n",
            )
            fn.writelines(nline)

    with open("cod.in", "w") as fn:
        for j in range(GS.shape[0]):
            nline = (
                str(ATOMS[j]),
                "   % .12f   % .12f   % .12f" % (CO_n[j, 0], CO_n[j, 1], CO_n[j, 2]),
                "\n",
            )
            fn.writelines(nline)

    os.chdir(path_old)
