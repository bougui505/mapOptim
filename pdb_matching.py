#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-10-12 09:38:56 (UTC+0200)

"""
Compute matching between structures
"""

import pymol.cmd as cmd
import numpy
import scipy.spatial.distance as dist


def load_pdb(pdbfile, obj, sel):
    cmd.load(pdbfile, object=obj)
    cmd.remove(f'(not ({obj} and {sel} and name CA)) and {obj}')
    coords = cmd.get_coords(selection=f'{obj}')
    return coords


def matching(A, B, dist_threshold=3.):
    """
    Matching between pair of coord set.
    Percentage of protein residues in the deposited model or the unique part of the deposited model that are within 3 A of a residue (residues represented by their CA) in the automatically-generated model.
    """
    cdist = dist.cdist(A, B)
    dmin = cdist.min(axis=0)
    n = B.shape[0]
    n_match = (dmin <= dist_threshold).sum()
    return n_match / n


def get_sequence(obj):
    seq = ''
    for chain in cmd.get_chains(obj):
        seq_ = cmd.get_fastastr(f'{obj} and chain {chain} and polymer.protein')
        seq_ = seq_.split()[1:]
        seq_ = ''.join(seq_)
        seq += seq_
    return seq


def seq_match(A, B, seq_A, seq_B, dist_threshold=3.):
    """
    Sequence matching between pair of coord set.
    Percentage of matching residues that have the same residue name.
    """
    cdist = dist.cdist(A, B)
    dmin = cdist.min(axis=0)
    assignment = cdist.argmin(axis=0)
    is_match = (dmin <= dist_threshold)
    n_match = 0
    for i, match in enumerate(is_match):
        if match:
            if seq_A[assignment[i]] == seq_B[i]:
                n_match += 1
    return n_match / is_match.sum()


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='Compute structural matching')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb1', help='first pdb file', type=str, required=True)
    parser.add_argument('--pdb2', help='second (reference) pdb file', type=str, required=True)
    parser.add_argument('--sel1', help='selection for pdb1 (default: all CA)', type=str, default='all')
    parser.add_argument('--sel2', help='selection for pdb2 (default: all CA)', type=str, default='all')
    args = parser.parse_args()
    A = load_pdb(args.pdb1, 'A_', args.sel1)
    A_chains = cmd.get_chains('A_')
    B = load_pdb(args.pdb2, 'B_', args.sel2)
    B_chains = cmd.get_chains('B_')
    print(f'PDB1: {A.shape[0]} CA')
    print(f'PDB2: {B.shape[0]} CA')
    matching = matching(A, B)
    print(f'% Matching:\t{matching*100:.1f}')
    seq_A = get_sequence('A_')
    seq_B = get_sequence('B_')
    seqmatching = seq_match(A, B, seq_A, seq_B)
    print(f'% Seq match:\t{seqmatching*100:.1f}')
