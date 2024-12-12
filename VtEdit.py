#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2024/02/22

    https://imagingsolution.net/program/python/tkinter/python_tkinter_image_viewer/ を参考に

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import logging

import numpy as np
from omegaconf import OmegaConf

import PySimpleGUI as sg

from MriImg import Mri
from VTShape import VTShape

import data_list

# ログの設定
logger = logging.getLogger(__name__)



class ImageCanvas:
    def __init__(self, cfg):
        self.sg_widget = sg.Canvas(size=(cfg.canvas.size_x, cfg.canvas.size_y))
        self.scale_val = cfg.mri.scale
        self.offset_x = cfg.mri.offset_x
        self.offset_y = cfg.mri.offset_y

        self.reset_transform()

    def get_sgwidget(self):
        return self.sg_widget

    def get_tkwidget(self):
        return self.sg_widget.TKCanvas

    def get_size(self):
        return self.sg_widget.TKCanvas.winfo_width(), self.sg_widget.TKCanvas.winfo_height()

    def reset_transform(self):
        '''アフィン変換を初期化（スケール１、移動なし）に戻す'''
        self.mat_affine = np.eye(3) # 3x3の単位行列

    def translate(self, offset_x, offset_y):
        ''' 平行移動 '''
        mat = np.eye(3) # 3x3の単位行列
        mat[0, 2] = float(offset_x)
        mat[1, 2] = float(offset_y)

        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale(self, scale:float):
        ''' 拡大縮小 '''
        mat = np.eye(3) # 単位行列
        mat[0, 0] = scale
        mat[1, 1] = scale

        self.mat_affine = np.dot(mat, self.mat_affine)


    def scale_at(self, scale:float, cx:float, cy:float):
        ''' 座標(cx, cy)を中心に拡大縮小 '''

        # 原点へ移動
        self.translate(-cx, -cy)
        # 拡大縮小
        self.scale(scale)
        # 元に戻す
        self.translate(cx, cy)


    def zoom_fit(self, mri_sz):
        '''画像をウィジェット全体に表示させる'''
        image_width, image_height = mri_sz
        # キャンバスのサイズ
        canvas_width, canvas_height = self.get_size()

        if image_width <= 0 or image_height <= 0 or canvas_width <= 0 or canvas_height <= 0:
            return

        # アフィン変換の初期化
        self.reset_transform()

        '''
        if (canvas_width * image_height) > (image_width * canvas_height):
            # ウィジェットが横長（画像を縦に合わせる）
            scale = canvas_height / image_height
            # あまり部分の半分を中央に寄せる
            offsetx = (canvas_width - image_width * scale) / 2
        else:
            # ウィジェットが縦長（画像を横に合わせる）
            scale = canvas_width / image_width
            # あまり部分の半分を中央に寄せる
            offsety = (canvas_height - image_height * scale) / 2
        '''
        # 拡大縮小
        self.translate(-self.offset_x, -self.offset_y)
        self.scale(self.scale_val)
        # あまり部分を中央に寄せる


def redraw(mri, vt, num, force_flg=False):
    if mri is not None:
        if force_flg or num != mri.num:
            mri.num=num
            mri.read_file()
            mri.draw()
    if vt is not None:
        if force_flg or num != vt.fr_num:
            vt.reset_para()
        vt.fr_num = num
        vt.draw()

def img_vt_reopen(cfg, mri, vt, fname, frnum):
    cfg.mri.fname = fname
    cfg.vt.fname = fname
    cfg.frame_num = frnum
    mri.num = cfg.frame_num
    mri.read_file()
    mri.draw()

    vt.fr_num = cfg.frame_num
    vt.reset_para()
    vt.draw()

def img_open(cfg, mri, vt):
    ret = sg.popup_get_text("set fname", default_text=cfg.mri.fname)
    cfg.mri.fname = ret
    cfg.vt.fname = ret
    cfg.frame_num = 0
    mri.num = cfg.frame_num
    mri.read_file()
    mri.draw()

    vt.fr_num = cfg.frame_num
    vt.delete_item()
    vt.read_dat()
    vt.draw()

def vt_open(cfg, vt):
    vtfn = cfg.vt.file.format(dir=cfg.vt.dir, spk=cfg.vt.spk, fname=cfg.vt.fname)
    fn = sg.popup_get_file('VTOpen', default_path=vtfn, file_types=(('vt data', '*.dat'),))
    if fn:
        vt.delete_item()
        vt.read_dat(fn)
        vt.draw()


def redraw_all(cfg, window, mri, vt, num, fname=None, ng_flg=False, dl_flg=False):
    img_vt_reopen(cfg, mri, vt, fname, num)
    window['NumIn'].update(f'{num}')
    if fname is not None:
        window['Fname'].update(fname)
    if dl_flg:
        window['dl_fnlst'].update(fname)
        window['dl_frlst'].update(f'{num}')
        ng_lbl = 'NG' if ng_flg else 'OK'
        window['dl_ng'].update(ng_lbl)

def make_save_filename(cfg, spk, fname):
    return cfg.vt.file.format(dir=cfg.vt.dir, spk=spk, fname=fname)

def vt_load(cfg, vt, spk, fname):
    if fname is not None:
        vt.delete_item()
        fn = cfg.vt.file.format(dir=cfg.vt.dir, spk=spk, fname=fname)
        vt.read_dat(fn)

def vt_x3(vt, mri):
        vt.dr_to_point()
        vt.select_all()
        vt.backup()
        for ii in range(3):
            vt.smoothing()
            vt.resample()
            vt.fine_tune(np.array(mri.pil_image))
        vt.smoothing()
        vt.resample()

def main(args, cfg):

    file_menu = ["File", ["Open", "VT open", "VT save as", "VT save"]]
    proc_menu =  ["Proc", ["Smoothing", "Resample", "FineTune", "FineTuneX3"]]
    parts_menu = ["PMenu", VTShape.AllList]

    num = cfg.canvas.frame_num

    canvas = ImageCanvas(cfg)
    layout = [[sg.ButtonMenu('File', file_menu), sg.Text(cfg.mri.fname, key='Fname'),
               sg.ButtonMenu('Proc', proc_menu),
               sg.Button('Undo'), sg.Button('<', key='Prev'),
               sg.InputText(str(num), size=(3,None), key='NumIn'), sg.Button('', key='Num'),
               sg.Button('>', key='Next'), sg.Button('All', key='AllSelect'),
               sg.ButtonMenu('', parts_menu, key='VtShapeMenu'),
               sg.Text(cfg.canvas.parts_name, key='VtShape')
               ],
              [canvas.get_sgwidget()]
              ]

    dlst = data_list.DataList(args.task)
    if len(dlst)>0:
        dlst_bar_flg = True
        nxt_fn_ix = dlst.stposi
        fr_num = dlst.body[dlst.stposi].num
        nxt_fr_ix = dlst.body[dlst.stposi].done_num
        num = nxt_fr_ix
        fnlst = dlst.get_fname_list()
        frlst = [str(i) for i in range(fr_num)]
        ng_flg = 'NG' if fr_num in dlst.body[dlst.stposi].ng_list else 'OK'
        low_panel = [sg.Combo(fnlst, default_value=fnlst[nxt_fn_ix], key='dl_fnlst', enable_events=True, readonly=True),
                     sg.Button('Prev', key='dl_prev'),
                     sg.Combo(frlst, default_value=frlst[nxt_fr_ix], key='dl_frlst', enable_events=True, readonly=True),
                     sg.OptionMenu(('OK','NG'), default_value=ng_flg, key='dl_ng'),
                     sg.Button('Next', key='dl_next'),
                     sg.Button('FTX3', key='dl_x3'),
                     sg.Button('Ok', key='dl_ok_next'),
                     sg.Button('Ng', key='dl_ng_next'),
                     sg.Button('Save', key='dl_save')
                     ]
        layout.append(low_panel)

        dlst_save_filename = args.otask
    else:
        dlst_bar_flg = False


    window = sg.Window("My Window", layout, finalize=True)
    if dlst_bar_flg:
        window['Fname'].update(fnlst[nxt_fn_ix])
        window['NumIn'].update(str(num))

    save_filename = None
    if dlst_bar_flg:
        cfg.mri.fname = fnlst[nxt_fn_ix]
    elif args.mri:
        cfg.mri.file = args.mri + '{frame_num:03d}.{ext}'
    mri = Mri(canvas, cfg.mri, num)
    canvas.zoom_fit(mri.get_size())
    mri.draw()

    if dlst_bar_flg:
        cfg.vt.fname = fnlst[nxt_fn_ix]
    elif args.vt:
        cfg.vt.file = args.vt
    vt = VTShape(canvas, cfg.vt, part=cfg.canvas.parts_name, fr_num=num, win=window)
    vt.draw()

    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        elif event == 'Undo':
            vt.undo()
        elif event == 'Next':
            vt.dr_to_point()
            num = num + 1 if num < len(vt)-1 else num
            redraw(mri, vt, num)
            window['NumIn'].update(f'{num}')
        elif event == 'Prev':
            vt.dr_to_point()
            num = num - 1 if num > 0 else 0
            redraw(mri, vt, num)
            window['NumIn'].update(f'{num}')
        elif event == 'Num':
            vt.dr_to_point()
            num = int(values['NumIn'])
            if num <0:
                num = 0
            if num > len(vt)-1:
                num = len(vt)-1
            redraw(mri, vt, num)
            window['NumIn'].update(f'{num}')
        elif event == 'AllSelect':
            vt.select_all()
        elif event == 'VtShapeMenu':
            vt.dr_to_point()
            vt.change_parts(values[event])
            window['VtShape'].update(values[event])
        elif event == 'LineSelect':
            vt.dr_to_point()
            vt.change_parts(values[event])
            window['VtShape'].update(values[event])
        elif event == 'dl_next':
            vt.dr_to_point()
            ng_flg = True if values['dl_ng'] == 'NG' else False
            cspk, cfname, cnum, cng_flg, cng = dlst.get_(num)
            spk, fname, num_, ng_flg, save_flg = dlst.get_next(ng_flg)
            if num_ > 0:
                num = num_
            if save_flg:
                save_filename = make_save_filename(cfg, cspk, cfname)
                vt.save_dat(save_filename)
                vt_load(cfg, vt, spk, fname)
                dlst.save(dlst_save_filename)
            redraw_all(cfg, window, mri, vt, num, fname, ng_flg, dlst_bar_flg)
        elif event == 'dl_prev':
            vt.dr_to_point()
            ng_flg = True if values['dl_ng'] == 'NG' else False
            cspk, cfname, cnum, cng_flg, cng = dlst.get_(num)
            spk, fname, num_, ng_flg, save_flg = dlst.get_prev(ng_flg)
            if num_>=0:
                num = num_
            if save_flg:
                save_filename = make_save_filename(cfg, cspk, cfname)
                vt.save_dat(save_filename)
                vt_load(cfg, vt, spk, fname)
                dlst.save(dlst_save_filename)
            redraw_all(cfg, window,mri,vt,num,fname,ng_flg, dlst_bar_flg)
        elif event == 'dl_ok_next':
            vt.dr_to_point()
            ng_flg = False
            cspk, cfname, cnum, cng_flg, cng = dlst.get_(num)
            spk, fname, num_, ng_flg, save_flg = dlst.get_next(ng_flg)
            if num_ > 0:
                num = num_
            if save_flg:
                save_filename = make_save_filename(cfg, cspk, cfname)
                vt.save_dat(save_filename)
                vt_load(cfg, vt, spk, fname)
                dlst.save(dlst_save_filename)
            redraw_all(cfg, window, mri, vt, num, fname, ng_flg, dlst_bar_flg)
            vt_x3(vt, mri)
        elif event == 'dl_ng_next':
            vt.dr_to_point()
            ng_flg = True
            cspk, cfname, cnum, cng_flg, cng = dlst.get_(num)
            spk, fname, num_, ng_flg, save_flg = dlst.get_next(ng_flg)
            if num_ > 0:
                num = num_
            if save_flg:
                save_filename = make_save_filename(cfg, cspk, cfname)
                vt.save_dat(save_filename)
                vt_load(cfg, vt, spk, fname)
                dlst.save(dlst_save_filename)
            redraw_all(cfg, window, mri, vt, num, fname, ng_flg, dlst_bar_flg)
            vt_x3(vt, mri)
        elif event == 'dl_frlst':
            num = int(values['dl_frlst'])
            vt.dr_to_point()
            dlst.set_(num)
            spk, fname, num, ng_flg, save_flg = dlst.get_(num)
            redraw_all(cfg, window, mri, vt, num, fname, ng_flg, dlst_bar_flg)
        elif event == 'dl_save':
            cspk, cfname, cnum, cng_flg, cng = dlst.get_(num)
            save_filename = make_save_filename(cfg, cspk, cfname)
            vt.save_dat(save_filename)
            dlst.save(dlst_save_filename)
        elif event =='dl_x3':
            vt_x3(vt, mri)

        elif event in values:
            if values[event] == 'Smoothing':
                vt.dr_to_point()
                vt.backup()
                vt.smoothing()
            elif values[event] == 'Resample':
                vt.dr_to_point()
                vt.backup()
                vt.resample()
            elif values[event] == 'FineTune':
                vt.dr_to_point()
                vt.backup()
                vt.fine_tune(np.array(mri.pil_image))
            elif values[event] == 'FineTuneX3':
                vt.dr_to_point()
                vt.select_all()
                vt.backup()
                for ii in range(3):
                    vt.smoothing()
                    vt.resample()
                    vt.fine_tune(np.array(mri.pil_image))
                vt.smoothing()
                vt.resample()
            elif values[event] == 'Open':
                img_open(cfg, mri, vt)
                window['Fname'].update(cfg.mri.fname)
            elif values[event] == 'VT open':
                vt_open(cfg, vt)
            elif values[event] == 'VT save as':
                vt.dr_to_point()
                vtfn = cfg.vt.file.format(dir=cfg.vt.dir, spk=cfg.vt.spk, fname=cfg.vt.fname)
                save_filename = sg.popup_get_file('VT', save_as=True, default_path=vtfn, file_types=(('vt data', '*.dat'),))
                if save_filename is not None:
                    vt.save_dat(save_filename)
            elif values[event] == 'VT save':
                vt.dr_to_point()
                if save_filename is None:
                    vtfn = cfg.vt.file.format(dir=cfg.vt.dir, spk=cfg.vt.spk, fname=cfg.vt.fname)
                    save_filename = sg.popup_get_file('VT', save_as=True, default_path=vtfn, file_types=(('vt data', '*.dat'),))
                if save_filename is not None:
                    vt.save_dat(save_filename)



    window.close()


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', '-c', default='config.yaml')
    parser.add_argument('--mri', default='')
    parser.add_argument('--vt', default='')
    parser.add_argument('-t', '--task', default='s1_all.csv')
    parser.add_argument('-o', '--otask', default='s1_all.csv')
    # parser.add_argument('-s', '--opt_str', default='')
    # parser.add_argument('--opt_int',type=int, default=1)
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

    cnf = OmegaConf.load(args.conf)

    main(args, cnf)
