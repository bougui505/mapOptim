#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-10-12 15:42:08 (UTC+0200)

import numpy
import hashlib
import pymol.cmd as cmd
import os


def md5sum(inp):
    inp = numpy.asarray(inp)
    inp = inp.flatten()
    instr = [str(e) for e in inp]
    instr = ''.join(instr)
    instr = instr.encode('utf-8')
    return hashlib.sha224(instr).hexdigest()


def get_sequence(chain):
    seq = cmd.get_fastastr(f'inpdb and chain {chain} and polymer.protein')
    seq = seq.split()[1:]
    seq = ''.join(seq)
    return seq


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Numpy array of shape (N,D) -- Point Cloud to Align (source)
        -    B: Numpy array of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = np.asarray([[1., 1.], [2., 2.], [1.5, 3.]])
        >>> R0 = np.asarray([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]])
        >>> B = (R0.dot(A.T)).T
        >>> t0 = np.array([3., 3.])
        >>> B += t0
        >>> B.shape
        (3, 2)
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.dot(A.T)).T + t
        >>> rmsd = np.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        2.5639502485114184e-16
        >>> B *= np.array([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.dot(A.T)).T + t
        >>> rmsd = np.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        2.5639502485114184e-16
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.dot(B_c)
    U, S, Vt = numpy.linalg.svd(H)
    V = Vt.T
    # Rotation matrix
    R = V.dot(U.T)
    # Translation vector
    t = b_mean - R.dot(a_mean)
    # rmsd
    A_aligned = (R.dot(A.T)).T + t
    rmsd = numpy.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
    # Get angles:
    theta_x = numpy.rad2deg(numpy.arctan2(R[2, 1], R[2, 2]))
    theta_y = numpy.rad2deg(numpy.arctan2(-R[2, 0], numpy.sqrt(R[2, 1]**2 + R[2, 2]**2)))
    theta_z = numpy.rad2deg(numpy.arctan2(R[1, 0], R[1, 1]))
    return R, t, rmsd, theta_x, theta_y, theta_z


def find_symmetry(seqhashes, chains, rmsd_threshold=3.):
    seqmatch = {h: [] for h in seqhashes}
    for h, c in zip(seqhashes, chains):
        seqmatch[h].append(c)
    for h in seqmatch:
        chains = seqmatch[h]
        if len(chains) > 1:
            chain1 = chains[0]
            B = cmd.get_coords(f'inpdb and chain {chain1} and name CA')
            for chain2 in chains[1:]:
                A = cmd.get_coords(f'inpdb and chain {chain2} and name CA')
                R, t, rmsd, theta_x, theta_y, theta_z = find_rigid_alignment(A, B)
                if rmsd <= rmsd_threshold:
                    print(f'{chain1}={chain2} (RMSD={rmsd:.2f}Å, θx={theta_x:.2f}°, θy={theta_y:.2f}°, θz={theta_z:.2f}°, tx={t[0]:.2f}Å, ty={t[1]:.2f}Å, tz={t[2]:.2f}Å)')
                    numpy.savez(f'symmetry_{chain1}-{chain2}.npz', R=R, t=t)


def apply_symmetry(R, t, outchain, outfilename):
    cmd.copy('symm', 'inpdb')
    coords = cmd.get_coords('symm')
    coords_symm = (R.dot(coords.T)).T + t
    cmd.load_coords(coords_symm, 'symm')
    myspace = {'outchain': outchain}
    cmd.alter('symm', 'chain=f"{outchain}"', space=myspace)
    cmd.save(outfilename, 'inpdb or symm')


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='Find exact symmetry in pdb file')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb', type=str, required=True, help='PDB file name')
    parser.add_argument('--select', type=str, help='Select part of the structure',
                        required=False, default='all')
    parser.add_argument('--apply', type=str, help='Apply the symmetry loaded from the given npy file')
    args = parser.parse_args()
    PDBFILENAME = args.pdb
    cmd.load(PDBFILENAME, 'inpdb')
    cmd.remove(f'not (inpdb and {args.select})')
    chains = cmd.get_chains('inpdb')
    seqhashes = []
    for chain in chains:
        seq = get_sequence(chain)
        seqhash = md5sum(seq)
        seqhashes.append(seqhash)
    if args.apply is None:
        find_symmetry(seqhashes, chains)
    else:
        symm = numpy.load(args.apply)
        R = symm['R']
        t = symm['t']
        outchain = os.path.basename(args.apply)
        outchain = os.path.splitext(outchain)[0]
        outchain = outchain.split('-')[-1]
        outfilename = f'{os.path.splitext(args.pdb)[0]}_sym.pdb'
        apply_symmetry(R, t, outchain, outfilename)
