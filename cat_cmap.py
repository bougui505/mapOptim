#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-10-08 11:06:10 (UTC+0200)

"""
Concatenate contact maps
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


npyfiles = sys.argv[1:]

cmaps = [np.load(npyfile) for npyfile in npyfiles]
lengths = [cmap.shape[0] for cmap in cmaps]
total_length = np.sum(lengths)

cmap_cat = - np.ones((total_length, total_length))
ind = 0
for cmap in cmaps:
    n = cmap.shape[0]
    cmap_cat[ind:ind + n, ind:ind + n] = cmap
    ind = ind + n

np.save('cmap_cat.npy', cmap_cat)
plt.matshow(cmap_cat)
plt.savefig('cmap_cat.png')
