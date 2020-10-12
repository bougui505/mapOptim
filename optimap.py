#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-09-29 15:48:44 (UTC+0200)

import pymol.cmd as cmd
import torch
import sys
try:
    from . import ICP
except ImportError:
    import ICP
import numpy


def print_progress(instr):
    sys.stdout.write(f'{instr}\r')
    sys.stdout.flush()


def get_cmap(coords, device, threshold=8., ca_switch=False, dist_ca=3.8, sigma_ca=3.):
    """
    - ca_switch: if True, apply a different distance threshold for consecutive CA
    - dist_ca: C-alpha - C-alpha distance
    """
    coords = coords.to(device)
    pdist = torch.cdist(coords, coords)
    S = torch.nn.Sigmoid()
    cmap_S = S(threshold - pdist)
    if ca_switch:
        n = coords.shape[0]
        A = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device))
        dist_to_diag = torch.abs(A[1] - A[0])
        cmap_G = torch.exp(-(pdist - dist_ca)**2 / (2 * sigma_ca**2))
        mask = dist_to_diag == 1
        cmap = torch.where(mask, cmap_G, cmap_S)
    else:
        cmap = cmap_S
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


def minimize(coords, cmap_ref, device, n_iter, do_normalize_P=False, coords_ref=None):
    n = coords.shape[0]
    # Permutation matrix
    P = torch.eye(n, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([P, ], lr=1e-3)
    n = coords.shape[0]
    for t in range(n_iter):
        optimizer.zero_grad()
        if do_normalize_P:
            P_norm = normalize_P(P, beta=0.1)
        else:
            P_norm = P
        coords_pred = permute(coords, P_norm)
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
    print("---")
    numpy.save('permutation.npy', P_norm.cpu().detach().numpy())
    return coords_pred


def read_fasta(fasta_file):
    """
    read only 1 chain
    """
    aa1 = list("ACDEFGHIKLMNPQRSTVWY")
    aa3 = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
    aa123 = dict(zip(aa1, aa3))
    # aa321 = dict(zip(aa3, aa1))
    with open(fasta_file) as fasta:
        seq = ''
        for line in fasta:
            if line[0] == '>':
                pass
            else:
                seq += line.strip()
    seq = [aa123[r] for r in seq]
    return seq


def write_pdb(obj, coords, outfilename, seq=None, resids=None):
    cmd.load_coords(coords, obj)
    if seq is not None:
        myspace = {}
        myspace['seq_iter'] = iter(seq)
        cmd.alter(obj, 'resn=f"{seq_iter.__next__()}"', space=myspace)
    if resids is not None:
        myspace = {}
        myspace['resid_iter'] = iter(resids)
        cmd.alter(obj, 'resi=f"{resid_iter.__next__()}"', space=myspace)
    cmd.save(outfilename, selection=obj)


def fix_coords_len(obj, offset, device):
    """
    Add or remove C-alpha if needed
    - offset > 0: add offset ca
    - offset < 0: remove offset ca
    """
    n = cmd.select(obj)
    if offset < 0:
        print(f'Removing {-offset} CA')
        # torm = numpy.random.choice(n, size=-offset, replace=False) + 1
        torm = numpy.arange(n)[::-1][:-offset]
        cmd.remove(f'{obj} and index {torm[-1]}-{torm[0]}')
    if offset > 0:
        print(f'Adding {offset} CA')
        pos = tuple(cmd.get_coords(obj).mean(axis=0))
        for i in range(offset):
            cmd.pseudoatom(obj, pos=pos, resn='ALA', hetatm=False, name='CA')
    coords_out = cmd.get_coords(obj)
    coords_out = torch.from_numpy(coords_out)
    coords_out = coords_out.to(device)
    return coords_out


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
    parser.add_argument('--seq', type=str, help='Fasta file with the sequence to write in the output pdb file')
    parser.add_argument('--resids', type=str, help='column text file with the residue numbering')
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    if args.seq is not None:
        seq = read_fasta(args.seq)
    else:
        seq = None

    if args.resids is not None:
        resids = numpy.genfromtxt(args.resids, dtype=int)
    else:
        resids = None

    if args.pdbref is not None:
        coords_ref = get_coords(args.pdbref, 'ref', device=device)
        cmap_ref = get_cmap(coords_ref, device=device)
        numpy.save(f'{os.path.splitext(args.pdbref)[0]}_cmap.npy', cmap_ref)
        sys.exit()
    cmap_ref = numpy.load(args.cmap)
    cmap_ref = torch.from_numpy(cmap_ref)
    cmap_ref = cmap_ref.float()
    cmap_ref = cmap_ref.to(device)
    coords_in = get_coords(args.pdb, 'mod', device)
    n = coords_in.shape[0]
    n_cmap = cmap_ref.shape[0]
    coords_in = fix_coords_len('mod', offset=n_cmap - n, device=device)
    if args.anchors is None:
        anchors = torch.clone(coords_in)
    else:
        anchors = get_coords(args.anchors, 'anchors', device)
    cmap_in = get_cmap(coords_in, device='cpu')
    n = coords_in.shape[0]
    coords_out = torch.clone(coords_in)
    for i in range(1):
        #  print(f'################ Iteration {i+1} ################')
        coords_out = minimize(coords_out, cmap_ref, device, args.niter)
        coords_out = ICP.icp(coords_out, anchors, device, 10, lstsq_fit_thr=1.9)
    cmap_out = get_cmap(coords_out, device='cpu').detach().numpy()
    coords_out = coords_out.cpu().detach().numpy()
    outpdbfilename = f"{os.path.splitext(args.pdb)[0]}_optimap.pdb"
    write_pdb(obj='mod', coords=coords_out, outfilename=outpdbfilename, seq=seq, resids=resids)
    plt.matshow(cmap_in.cpu().numpy())
    plt.savefig('cmap_in.png')
    plt.matshow(cmap_ref.cpu().numpy())
    plt.savefig('cmap_ref.png')
    plt.matshow(cmap_out)
    plt.savefig('cmap_out.png')
