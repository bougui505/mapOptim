#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-10-07 15:05:21 (UTC+0200)

"""
Convert a Casp contact map to a npy file
See: https://predictioncenter.org/casp14/index.cgi?page=format#RR
for the input file format
"""

import os
import argparse
import numpy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Convert a Casp contact map to a npy file. See: https://predictioncenter.org/casp14/index.cgi?page=format#RR for the input file format')
parser.add_argument('--cmap', type=str, help='CASP contact map file', required=True)
parser.add_argument('--sel', type=str, help='Residue selection for the contact map (e.g.: 10-24+30-65+70-94)')
parser.add_argument('--ss', type=str, help='Secondary structure prediction file to fill 4-first diagonal (optional). The format is e.g.: 3 I H   0.985 0.000 0.014, with the resid, resname, SS-type, H-propensity, E-propensity and C-propensity as in DSSP')
args = parser.parse_args()


def parse_selection(selection_string):
    sels = selection_string.split('+')
    resids = []
    for chunk in sels:
        start, stop = chunk.split('-')
        start = int(start)
        stop = int(stop)
        resids.extend(numpy.arange(start, stop + 1))
    return resids


def read_ss(infile):
    data = numpy.genfromtxt(infile, dtype=str)
    resids = numpy.int_(numpy.float_(data[:, 0]))
    hec = numpy.float_(data[:, 3:])  # propensity for H, E and C class as in DSSP
    ss_mapping = dict(zip(resids, hec))
    return ss_mapping


col1 = numpy.genfromtxt(args.cmap, usecols=(0,), dtype=str)
start_ind = numpy.where(col1 == 'MODEL')[0][0]
stop_ind = numpy.where(col1 == 'END')[0][0]
data = numpy.genfromtxt(args.cmap, usecols=(0, 1, 2), skip_header=start_ind + 1, skip_footer=len(col1) - stop_ind)
min_resid1, min_resid2, min_prob = data.min(axis=0)
max_resid1, max_resid2, max_prob = data.max(axis=0)
min_resid = int(min(min_resid1, min_resid2))
max_resid = int(max(max_resid1, max_resid2))
if args.sel is None:
    sel = range(min_resid, max_resid + 1)
else:
    sel = parse_selection(args.sel)
n = len(sel)
cmap = numpy.eye(n)
mapping = dict(zip(sel, range(len(sel))))

for d in data:
    r1, r2, p = d
    if r1 in sel and r2 in sel:
        r1 = int(r1)
        r2 = int(r2)
        ind1 = mapping[r1]
        ind2 = mapping[r2]
        cmap[ind1, ind2] = p
        cmap[ind2, ind1] = p

# alpha-helices prediction
if args.ss is not None:
    ss = read_ss(args.ss)
    for r in sel:
        for rnext in [r + 2, r + 3, r + 4]:
            if rnext in sel:
                cmap[mapping[r], mapping[rnext]] = ss[rnext][0]
                cmap[mapping[rnext], mapping[r]] = ss[rnext][0]

# First diagonal (topology):
for r in sel:
    if r + 1 in sel:
        cmap[mapping[r], mapping[r + 1]] = 1.
        cmap[mapping[r + 1], mapping[r]] = 1.
    else:
        ind = mapping[r]
        if ind + 1 < n:
            cmap[ind, ind + 1] = 0.
            cmap[ind + 1, ind] = 0.

print(f'Contact map shape: {cmap.shape}')
outbasename = os.path.split(os.path.splitext(args.cmap)[0])[1]
numpy.save(f'{outbasename}.npy', cmap)
numpy.save(f'{outbasename}.top.npy', cmap.diagonal(offset=1))
plt.matshow(cmap)
plt.colorbar()
plt.savefig(f'{outbasename}.png')
