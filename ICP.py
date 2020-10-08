#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-10-02 10:00:15 (UTC+0200)

import sys
import torch


def print_progress(instr):
    sys.stdout.write(f'{instr}\r')
    sys.stdout.flush()


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def transform(coords, R, t):
    """
    Apply R and a translation t to coords
    """
    coords_out = R.mm(coords.T).T + t
    return coords_out


def get_RMSD(A, B):
    """
    Return the RMSD between the two set of coords
    """
    rmsd = torch.sqrt(((A - B)**2).sum(axis=1).mean())
    return rmsd


def assign_anchors(coords, coords_ref, dist_thr=None):
    """
    Assign the closest anchors with coords coords_ref
    >>> coords_ref = torch.tensor([[0., 0., 0.], [1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> coords = torch.zeros_like(coords_ref)
    >>> coords[0] = coords_ref[1]
    >>> coords[1] = coords_ref[3]
    >>> coords[2] = coords_ref[0]
    >>> coords[3] = coords_ref[2]
    >>> get_RMSD(coords, coords_ref)
    tensor(7.5166)
    >>> assignment = assign_anchors(coords, coords_ref)
    >>> assignment
    tensor([2, 0, 3, 1])
    >>> coords_ordered = coords[assignment]
    >>> get_RMSD(coords_ordered, coords_ref)
    tensor(0.)
    >>> coords[2] += 100.
    >>> assignment, sel = assign_anchors(coords, coords_ref, dist_thr=4.)
    >>> assignment
    tensor([0, 3, 1])
    >>> sel
    tensor([1, 2, 3])
    >>> coords_ordered = coords[assignment]
    >>> get_RMSD(coords_ordered, coords_ref[sel])
    tensor(0.)
    """
    cdist = torch.cdist(coords,
                        coords_ref)
    mindists, argmins = torch.min(cdist, axis=1)
    order = mindists.argsort()
    if dist_thr is not None:
        sel = mindists[order] <= dist_thr
        order = order[torch.nonzero(sel, as_tuple=True)]
    maxval = cdist.max()
    assignment = - torch.ones_like(mindists, dtype=torch.long)
    for i in order:
        j = torch.argmin(cdist[i])
        assignment[j] = i
        cdist[:, j] = maxval * 10.
    if dist_thr is not None:
        sel = (assignment != -1)
        sel = torch.nonzero(sel, as_tuple=True)[0]
        assignment = assignment[sel]
        return assignment.squeeze(), sel
    else:
        return assignment


def find_initial_alignment(coords, coords_ref, fsize=30):
    """
    - fsize: fragment size
    """
    chunks = torch.split(coords, fsize)[:-1]
    chunks_ref = torch.split(coords_ref, fsize)[:-1]
    n_chunks = len(chunks)
    rmsd_min = 9999.99
    for i, chunk in enumerate(chunks):
        for j, chunk_ref in enumerate(chunks_ref):
            R, t = find_rigid_alignment(chunk, chunk_ref)
            chunk_aligned = transform(chunk, R, t)
            # coords_aligned = transform(coords, R, t)
            rmsd = get_RMSD(chunk_aligned, chunk_ref)
            # rmsd = get_RMSD(coords_aligned, coords_ref)
            if rmsd < rmsd_min:
                rmsd_min = rmsd
                i_best, j_best = i, j
                R_best, t_best = R, t
    # print(rmsd_min, i_best, j_best)
    return R_best, t_best


def icp(coords, coords_ref, device, n_iter, dist_thr=3.8, lstsq_fit_thr=0.):
    """
    Iterative Closest Point
    - lstsq_fit_thr: distance threshold for least square fit (if 0: no lstsq_fit)
    """
    coords_out = coords.detach().clone()
    R, t = find_initial_alignment(coords_out, coords_ref)
    coords_out = transform(coords_out, R, t)
    assignment, sel = assign_anchors(coords_ref, coords_out, dist_thr=dist_thr)
    rmsd = get_RMSD(coords_ref[assignment], coords_out[sel])
    n_assigned = len(sel)
    print(f"Initial RMSD: {rmsd} Å; n_assigned: {n_assigned}/{len(coords)} at less than {dist_thr} Å")
    for i in range(n_iter):
        R, t = find_rigid_alignment(coords_out[sel], coords_ref[assignment])
        coords_out = transform(coords_out, R, t)
        assignment, sel = assign_anchors(coords_ref, coords_out, dist_thr=dist_thr)
        rmsd = get_RMSD(coords_out[sel], coords_ref[assignment])
        n_assigned = len(sel)
        print_progress(f'{i+1}/{n_iter}: {rmsd} Å; n_assigned: {n_assigned}/{len(coords)} at less than {dist_thr} Å             ')
    sys.stdout.write('\n')
    print("---")
    if lstsq_fit_thr > 0.:
        coords_out = lstsq_fit(coords_out, coords_ref, dist_thr=lstsq_fit_thr)
        assignment, sel = assign_anchors(coords_ref, coords_out, dist_thr=dist_thr)
        rmsd = get_RMSD(coords_out[sel], coords_ref[assignment])
        print(f'lstsq_fit: {rmsd} Å; n_assigned: {n_assigned}/{len(coords)} at less than {dist_thr} Å')
    sys.stdout.write('\n')
    return coords_out


def lstsq_fit(coords, coords_ref, dist_thr=1.9, ca_dist=3.8):
    """
    Perform a least square fit of coords on coords_ref
    """
    n = coords.shape[0]
    coords_out = torch.clone(coords)
    device = coords_out.device
    coords_out = coords_out.to('cpu')
    assignment, sel = assign_anchors(coords_ref, coords, dist_thr=dist_thr)
    # Not yet implemented on gpu so go to cpu:
    if coords_ref.is_cuda:
        coords_ref = coords_ref.to('cpu')
    if coords.is_cuda:
        coords = coords.to('cpu')
    # Topology
    anchors = coords_ref[assignment]
    pdist = torch.cdist(anchors, anchors)
    sequential = torch.diagonal(pdist, offset=1)
    sigma_ca = 0.1
    topology = torch.exp(-(sequential - ca_dist) / (2 * sigma_ca**2))
    toposel = torch.nonzero(topology > .5, as_tuple=True)[0]
    sel = sel[toposel]
    assignment = assignment[toposel]
    ##########
    X, _ = torch.lstsq(coords_ref[assignment].T, coords[sel].T)
    coords_out[sel] = (coords[sel].T.mm(X[:n])).T
    n_assigned = len(sel)
    print(f"lstsq_fit: n_assigned: {n_assigned}/{n} at less than {dist_thr} Å")
    coords_out = coords_out.to(device)
    return coords_out


if __name__ == '__main__':
    import pymol.cmd as cmd
    import optimap
    import numpy as np  # For doctest
    import doctest
    import os
    import argparse

    doctest.testmod()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description='Iterative Closest Point algorithm for structural alignment')
    parser.add_argument('--pdb1', type=str, help='First protein structure (mobile)',
                        required=True)
    parser.add_argument('--pdb2', type=str, help='Second protein structure (reference)',
                        required=True)
    parser.add_argument('--niter', type=int, help='Number of iterations (default: 100)',
                        default=100)
    parser.add_argument('--flex', type=float, help='Distance threshold for flexible fitting using least square (default=0, no least square flexible fitting)', default=0.)
    args = parser.parse_args()

    coords_ref = optimap.get_coords(args.pdb2, 'ref', device=device)
    coords_in = optimap.get_coords(args.pdb1, 'mod', device)
    # Try to align
    # R, t = find_rigid_alignment(coords_in, coords_ref)
    # coords_out = transform(coords_in, R, t)
    # rmsd = get_RMSD(coords_out, coords_ref)
    # print(f'RMSD for rigid alignment: {rmsd}')
    # coords_out = coords_out.cpu().detach().numpy()
    # cmd.load_coords(coords_out, 'mod')
    # cmd.save('out_align.pdb', selection='mod')
    # Try the ICP
    coords_out = icp(coords_in, coords_ref, device, args.niter, lstsq_fit_thr=args.flex)
    coords_out = coords_out.cpu().detach().numpy()
    cmd.load_coords(coords_out, 'mod')
    cmd.save(f'{os.path.splitext(args.pdb1)[0]}_icp.pdb', selection='mod')
