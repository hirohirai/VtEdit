#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2024/08/13

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import numpy as np
import MriImg
import VTShape

# ログの設定
logger = logging.getLogger(__name__)


def fine_tuneX3(part, img_np):
    flip_flg = True if part.name in ['uplips', 'lowerlips', 'uppharynx', 'lowpharynx'] else False
    for ii in range(3):
        part.smoothing()
        part.resample()
        part.fine_tune(img_np, flip_flg)
    part.smoothing()
    part.resample()


def main(args):
    part_names = args.parts.split(',')
    vts = VTShape.VTShapeBase(part_names[0])
    vts.read_dat(args.idat)
    if args.ix >= 0:
        st = args.ix
        ed = args.ix+1
    else:
        st = args.st
        ed = args.ed
        if ed<0:
            ed = len(vts)
    for ix in range(st, ed):
        fname = f'{args.mri}/{ix:03}.npy'
        mri = MriImg.MriBase(fname)
        img_np = np.array(mri.pil_image)
        for pname in part_names:
            part = vts.parts[pname]
            part.fr_num=ix
            fine_tuneX3(part,img_np)


    odir = os.path.split(args.odat)[0]
    os.makedirs(odir, exist_ok=True)
    vts.save_dat(args.odat)

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri', default='data/Img/s1/A01')
    #parser.add_argument('--file_part', default='all')
    parser.add_argument('--idat', default='data/dat/s1/all/A01.dat')
    parser.add_argument('--odat', default='odata/dat/s1/all/A01.dat')
    parser.add_argument('--ix', type=int, default=-1)
    parser.add_argument('--st', type=int, default=0)
    parser.add_argument('--ed', type=int, default=-1)
    parser.add_argument('--parts', default='tongue,uplips,lowerlips,palate,uppharynx,lowpharynx,larynx')
    # parser.add_argument('-i', '--input',type=argparse.FileType('r'), default='-')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--debug', '-d', action='store_true')
    parser.add_argument('--log', default='')
    args = parser.parse_args()

    if args.debug:
        if args.log:
            logging.basicConfig(filename=args.log, level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.DEBUG)
    elif args.verbose:
        if args.log:
            logging.basicConfig(filename=args.log, level=logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO)

    main(args)
