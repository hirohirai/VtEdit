#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2024/12/09

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging


# ログの設定
logger = logging.getLogger(__name__)

class DataFnFrame:
    def __init__(self, spk, fname=None, frnum=None, nextp=0, ng=[]):
        if fname is None:
            ee = spk.strip().split(',')
            self.spk = ee[0]
            self.fname = ee[1]
            self.num = int(ee[2])
            self.posi = int(ee[3])
            self.done_num = self.posi
            if ee[4] == '':
                self.ng_list = []
            else:
                self.ng_list = [int(x) for x in ee[4:]]
        else:
            self.spk = spk
            self.fname = fname
            self.num = frnum
            self.posi = nextp
            self.done_num = self.posi
            self.ng_list = ng

    def __str__(self):
        ng = ','.join([str(x) for x in self.ng_list])
        return f'{self.spk},{self.fname},{self.num},{self.done_num},{ng}'

    def get_next(self, ng_flg=None):
        if ng_flg is not None:
            if ng_flg == True:
                if self.posi not in self.ng_list:
                    self.ng_list.append(self.posi)
            else:
                if self.posi in self.ng_list:
                    self.ng_list.remove(self.posi)
        self.posi += 1
        if self.posi>self.done_num:
            self.done_num = self.posi
            if self.done_num > self.num:
                self.done_num = self.num

        if self.posi >= self.num:
            self.posi = self.num-1
            return -1

        return self.posi

    def get_prev(self, ng_flg=None):
        if ng_flg is not None:
            if ng_flg == True:
                if self.posi not in self.ng_list:
                    self.ng_list.append(self.posi)
            else:
                if self.posi in self.ng_list:
                    self.ng_list.remove(self.posi)
        self.posi -= 1
        if self.posi < 0:
            self.posi = 0
            return -1
        return self.posi

    def is_ng(self, ix):
        return ix in self.ng_list




def get_frame_num(fn):
    ix = 0
    with open(fn) as ifs:
        for ix,ll in enumerate(ifs, 1):
            if len(ll.strip())==0:
                ix = ix -1
                break
    return ix

class DataList:
    def __init__(self, fn, dat_dir=None):
        if dat_dir:
            self.body = []
            with open(fn) as ifs:
                for lin in ifs:
                    spk,fname = lin.strip().split(',')
                    frnum = get_frame_num(f'{dat_dir}/{spk}/all/{fname}.dat')
                    self.body.append(DataFnFrame(spk,fname,frnum))
                    
            self.stposi=0
        else:
            self.body = []
            self.stposi = -1
            if os.path.isfile(fn):
                with open(fn) as ifs:
                    for ix, lin in enumerate(ifs):
                        dff = DataFnFrame(lin)
                        if self.stposi<0 and dff.done_num < dff.num:
                            self.stposi = ix
                        self.body.append(dff)

    def save(self, fn):
        with open(fn, 'w') as ofs:
            for elm in self.body:
                print(elm, file=ofs)

    def set_(self, ix):
        self.body[self.stposi].posi = ix

    def get_(self,ix):
        return self.body[self.stposi].spk,self.body[self.stposi].fname,ix,self.body[self.stposi].is_ng(ix), False

    def get_next(self, ng_flg=None):
        n_frm = self.body[self.stposi].get_next(ng_flg)
        save_flg = False
        while n_frm < 0:
            save_flg = True
            self.stposi += 1
            if self.stposi >= len(self.body):
                self.stposi = len(self.body)-1
                return None, None, -1, False, save_flg
            n_frm = self.body[self.stposi].get_next()

        return self.body[self.stposi].spk,self.body[self.stposi].fname,n_frm,self.body[self.stposi].is_ng(n_frm), save_flg

    def get_prev(self, ng_flg=None):
        p_frm = self.body[self.stposi].get_prev(ng_flg)
        save_flg = False
        if p_frm < 0:
            save_flg = True
            self.stposi -= 1
            if self.stposi < 0:
                self.stposi = 0
                return None, None, -1, False, save_flg
            p_frm = self.body[self.stposi].done_num-1
            if p_frm<0:
                p_frm = 0

        return self.body[self.stposi].spk,self.body[self.stposi].fname,p_frm, self.body[self.stposi].is_ng(p_frm), save_flg

    def __len__(self):
        return len(self.body)

    def get_fname_list(self):
        fnl = [ee.fname for ee in self.body]
        return fnl



def main(args):
    dl = DataList(args.ifile, args.dat_dir)
    dl.save(args.ofile)



if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('ifile')
    parser.add_argument('ofile')
    parser.add_argument('--dat_dir', default='../DBS_/rtmri-atr503/dat')
    # parser.add_argument('--opt_int',type=int, default=1)
    # parser.add_argument('-o', '--output',type=argparse.FileType('w'), default='-')
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
