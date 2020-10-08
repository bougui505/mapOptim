#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-09-29 15:48:44 (UTC+0200)

import pymol.cmd as cmd
import torch
import sys
import ICP
import numpy


def print_progress(instr):
    sys.stdout.write(f'{instr}\r')
    sys.stdout.flush()


def get_cmap(coords, device, threshold=8., dist_ca=3.8, sigma_ca=.1):
    """
    - dist_ca: C-alpha - C-alpha distance
    """
    n = coords.shape[0]
    A = torch.meshgrid(torch.arange(n), torch.arange(n))
    dist_to_diag = torch.abs(A[1] - A[0])
    dist_to_diag = dist_to_diag.to(device)
    pdist = torch.cdist(coords, coords)
    S = torch.nn.Sigmoid()
    cmap_S = S(threshold - pdist)
    cmap_G = torch.exp(-(pdist - dist_ca)**2 / (2 * sigma_ca**2))
    cmap = torch.where(dist_to_diag == 1, cmap_G, cmap_S)
    cmap = cmap.to(device)
    return cmap


def get_coords(pdbfilename, object, device, selection=None):
    if selection is None:
        selection = f'{object} and name CA'
    cmd.load(pdbfilename, object=object)
    cmd.remove(f'(not name CA) and {object}')
    coords = cmd.get_coords(selection=selection)
    coords = torch.from_numpy(coords)
    coords = coords.to(device)
    return coords


def permute(coords, weights):
    out = coords.t().mm(weights).t()
    # out = weights.mm(coords)
    # out = coords.t().mm(torch.nn.functional.softmax(weights, dim=1)).t()
    return out


def minsum(v, axis=1, n=2., eps=1e-6):
    """
    A sum over v that returns a value close to the minima
    """
    w = (1 / (1 / (v + eps) ** n).sum(axis=axis)) ** (1. / n)
    return w


def anchor_loss(coords, anchors):
    cdist = torch.cdist(coords - coords.mean(axis=0), anchors - anchors.mean(axis=0))
    mindists = torch.min(cdist, axis=1)[0]
    # mindists = minsum(cdist, axis=1)
    loss = (mindists**2).mean()
    return loss


def cmap_loss(cmap_pred, cmap_true, w0=0.05):
    cmap_pred = cmap_pred.flatten()
    cmap_true = cmap_true.flatten()
    bceloss = torch.nn.BCELoss(weight=(cmap_true + w0 * torch.ones_like(cmap_true)))
    # bceloss = torch.nn.BCELoss(weight=cmap_true)
    output = bceloss(cmap_pred, cmap_true)
    return output


def normalize_P(P_in, beta):
    P_out = torch.exp(beta * P_in)
    P_1 = P_out / P_out.sum(dim=0)
    P_out = P_1 / P_1.sum(dim=1)
    return P_out


def is_normed(P, eps=1e-3):
    return torch.abs(P.sum(dim=0) - 1.).max() < eps and torch.abs(P.sum(dim=1) - 1.).max() < eps


def entropy_loss(P, eps=1e-3):
    loss = -(P * torch.log(P + eps)).mean()
    return loss


def get_rmsd(A, B):
    rmsd = torch.sqrt(((A - B)**2).sum(axis=1).mean())
    return rmsd


def minimize(coords, cmap_ref, device, n_iter, coords_ref=None):
    n = coords.shape[0]
    # Permutation matrix
    P = torch.eye(n, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([P, ], lr=1e-3)
    n = coords.shape[0]
    for t in range(n_iter):
        optimizer.zero_grad()
        coords_pred = permute(coords, P)
        cmap_pred = get_cmap(coords_pred, device=device)
        loss = cmap_loss(cmap_pred, cmap_ref)
        loss.backward()
        optimizer.step()
        if t % 100 == 99:
            if coords_ref is not None:
                rmsd = get_rmsd(coords_pred, coords_ref)
                print_progress(f'{t+1}/{n_iter}: L={loss}, rmsd={rmsd}')
            else:
                print_progress(f'{t+1}/{n_iter}: L={loss}')
    sys.stdout.write('\n')
    # P_norm = normalize_P(P, beta=betas[t])
    numpy.save('permutation.npy', P.cpu().detach().numpy())
    return coords_pred


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    import argparse

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")

    parser = argparse.ArgumentParser(description='Model optimization based on contact map')
    parser.add_argument('--pdb', type=str, help='Protein structure to optimize')
    parser.add_argument('--anchors', type=str, help='PDB file containing the coordinates to anchor the model on. If not given, the pdb file given with the --pdb option is taken.')
    parser.add_argument('--cmap', type=str, help='npy file of the contact map')
    parser.add_argument('--niter', type=int, help='Number of iteration for optimizer (default: 10000)',
                        default=10000)
    parser.add_argument('--pdbref', type=str, help='Generate a npy file with the contact map build from the pdb and exit')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    if args.pdbref is not None:
        coords_ref = get_coords(args.pdbref, 'ref', device=device)
        cmap_ref = get_cmap(coords_ref, device=device)
        numpy.save(f'{os.path.splitext(args.pdbref)[0]}_cmap.npy', cmap_ref)
        sys.exit()
    coords_in = get_coords(args.pdb, 'mod', device)
    if args.anchors is None:
        anchors = torch.clone(coords_in)
    else:
        anchors = get_coords(args.anchors, 'anchors', device)
    cmap_ref = numpy.load(args.cmap)
    cmap_ref = torch.from_numpy(cmap_ref)
    cmap_ref = cmap_ref.float()
    cmap_ref = cmap_ref.to(device)
    cmap_in = get_cmap(coords_in, device='cpu')
    n = coords_in.shape[0]
    coords_out = minimize(coords_in, cmap_ref, device, args.niter)
    coords_out = ICP.icp(coords_out, anchors, device, 10)
    cmap_out = get_cmap(coords_out, device='cpu').detach().numpy()
    coords_out = coords_out.cpu().detach().numpy()
    cmd.load_coords(coords_out, 'mod')
    cmd.save('out.pdb', selection='mod')
    plt.matshow(cmap_in.cpu().numpy())
    plt.savefig('cmap_in.png')
    plt.matshow(cmap_ref.cpu().numpy())
    plt.savefig('cmap_ref.png')
    plt.matshow(cmap_out)
    plt.savefig('cmap_out.png')
