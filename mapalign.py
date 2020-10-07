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


def get_cmap(coords, device, threshold=8.):
    pdist = torch.cdist(coords, coords)
    S = torch.nn.Sigmoid()
    cmap = S(threshold - pdist)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    coords_ref = get_coords('5v6p_.pdb', 'ref', device=device)
    cmap_ref = get_cmap(coords_ref, device=device)
    # cmap_ref[cmap_ref < 0.5] = 0.
    # cmap_ref[cmap_ref >= 0.5] = 1.
    coords_in = get_coords('map_to_model_5v6p_8637_.pdb', 'mod', device)
    cmap_in = get_cmap(coords_in, device='cpu')
    n = coords_in.shape[0]
    coords_out = minimize(coords_in, cmap_ref, device, 10000)
    coords_out = ICP.icp(coords_out, coords_in, device, 10)
    rmsd = torch.sqrt(((coords_out - coords_ref)**2).sum(axis=1).mean())
    print(f'RMSD to deposited structure: {rmsd}')
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
