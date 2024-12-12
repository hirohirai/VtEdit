#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

    Author: hirai
    Data: 2024/02/28

"""
import sys, os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import logging

import numpy as np
import scipy
import scipy.signal as signal

# ログの設定
logger = logging.getLogger(__name__)

class VtPoint:
    Circle_R = 2
    Select_C = 'red'
    UnSelect_C = 'blue'
    def __init__(self, ix, dr_posi, select_func, move_func, tkcanvas):
        self.ix = ix
        self.dr_posi = dr_posi  # np.array
        self.select_func = select_func
        self.move_func = move_func
        self.selected = False
        self.tkcanvas = tkcanvas
        self.tk_id = self.draw_circle()

    def reset_para(self):
        self.selected = False

    def draw_circle(self):
        if self.selected:
            col=self.Select_C
        else:
            col=self.UnSelect_C

        id_ = self.tkcanvas.create_oval(self.dr_posi[0]-self.Circle_R, self.dr_posi[1]-self.Circle_R,
                                         self.dr_posi[0]+self.Circle_R, self.dr_posi[1]+self.Circle_R,
                                         fill=col, activefill='yellow')
        self.tkcanvas.tag_bind(id_, '<Button-1>', self.click_func)
        self.tkcanvas.tag_bind(id_, '<ButtonRelease-1>', self.release_func)
        self.tkcanvas.tag_bind(id_, '<Button1-Motion>', self.drag_func)
        return id_

    def select(self):
        self.selected = True
        self.tkcanvas.itemconfig(self.tk_id, fill=self.Select_C)

    def unselect(self):
        self.selected = False
        self.tkcanvas.itemconfig(self.tk_id, fill=self.UnSelect_C)

    def move(self, dxy):
        self.dr_posi = self.dr_posi + dxy
        self.tkcanvas.move(self.tk_id, dxy[0], dxy[1])

    def redraw(self):
        self.tkcanvas.delete(self.tk_id)
        self.tk_id = self.draw_circle()

    def click_func(self, ev):
        self.move_flg=False
        self.last_p = np.array([ev.x, ev.y])
        if not self.selected:
            if ev.state == 1:
                self.select_func('set_s', self.ix)
            else:
                self.select_func('add', self.ix)
        else:
            self.move_func('st', self.ix, ev)
    def drag_func(self, ev):
        if self.selected:
            self.move_flg=True
            xy = np.array([ev.x, ev.y])
            dxy = xy - self.last_p
            self.last_p = xy

            self.move_func('mv', self.ix, ev, dxy)

    def release_func(self, ev):
        if self.move_flg:
            self.move_func('rd', self.ix, ev)

    def delete_item(self):
        self.tkcanvas.delete(self.tk_id)


def smoothing(point, st, ed):
    if ed - st + 1 > 5:
        dat = signal.savgol_filter(point[st:ed+1], 5, 2, axis=0)
        point[st:ed+1] = dat
    return point


def resample(point, resample_num):
    nn = len(point)
    f = scipy.interpolate.interp1d(np.arange(nn) * resample_num, point.T)
    y = f(np.arange((nn-1)*resample_num)).T
    dxy = y[1:] - y[:-1]
    dxy2 = dxy * dxy
    dd = np.sqrt(dxy2[:,0] + dxy2[:,1])
    ddcumsum = np.cumsum(dd)
    po = np.arange(nn) * ddcumsum[-1] /(nn-1)

    pix =[0]
    for ix in range(1,nn-1):
        dis = np.abs(ddcumsum - po[ix])
        pix.append(np.argmin(dis))
    pix.append(-1)
    return y[np.array(pix)]


class VtPartsBase:
    FineTuneDis = 3
    FineTuneNum = 10

    def __init__(self, name='tongue', dat=None):
        self.name = name
        self.point = self.set_data(dat)   # ポイントのデータ 全てのフレームを含む
        self.fr_num = 0

        self.resample_num = 100 # 何倍にするか

    def __len__(self):
        return len(self.point)

    def set_data(self, dat):
        if len(dat.shape) == 2:
            sz = dat.shape
            return dat.reshape((sz[0], int(sz[1]/2),2))
        else:
            return dat

    def smoothing(self):
        self.point[self.fr_num] = smoothing(self.point[self.fr_num], 0, len(self.point[self.fr_num])-1)

    def resample(self):
        self.point[self.fr_num] = resample(self.point[self.fr_num], self.resample_num)

    def fine_tune_base(self, img_np, flip_flg):
        sz = img_np.shape
        f = scipy.interpolate.interp2d(np.arange(sz[0]),np.arange(sz[1]), img_np, 'cubic')
        c_dat = self.point[self.fr_num]
        ret_xy = [c_dat[0]]
        for ix in range(1,len(c_dat)-1):
            a0 = (c_dat[ix + 1][0] - c_dat[ix - 1][0])
            a1 = (c_dat[ix + 1][1] - c_dat[ix - 1][1])
            a = -a0 / (a1 + 0.00000001)
            b = c_dat[ix][1] - a * c_dat[ix][0]
            dl = np.sqrt(1+a*a)
            if a0 < 0 and a1 < 0:
                xi = np.arange(self.FineTuneNum, -(self.FineTuneNum + 1), -1) / self.FineTuneNum * (
                            self.FineTuneDis / dl) + c_dat[ix][0]
                yi = a * xi + b
                imtmp = f(xi, yi)
                y = np.array([imtmp[ii, self.FineTuneNum*2-ii] for ii in range(self.FineTuneNum*2)])
                dy = y[1:] - y[:-1]
                if flip_flg:
                    kk = np.argmax(dy)
                else:
                    kk = np.argmin(dy)
            elif a0 > 0 and a1 < 0:
                xi = np.arange(self.FineTuneNum, -(self.FineTuneNum + 1), -1) / self.FineTuneNum * (
                            self.FineTuneDis / dl) + c_dat[ix][0]
                yi = a * xi + b
                imtmp = f(xi, yi)
                y = np.array([imtmp[self.FineTuneNum*2-ii, self.FineTuneNum*2-ii] for ii in range(self.FineTuneNum*2)])
                dy = y[1:] - y[:-1]
                if flip_flg:
                    kk = np.argmax(dy)
                else:
                    kk = np.argmin(dy)
            elif a0 > 0 and a1 > 0:
                xi = np.arange(-self.FineTuneNum, self.FineTuneNum + 1, 1) / self.FineTuneNum * (
                            self.FineTuneDis / dl) + c_dat[ix][0]
                yi = a * xi + b
                imtmp = f(xi, yi)
                y = np.array([imtmp[self.FineTuneNum*2 - ii, ii] for ii in range(self.FineTuneNum*2)])
                dy = y[1:] - y[:-1]
                if flip_flg:
                    kk = np.argmax(dy)
                else:
                    kk = np.argmin(dy)
            else:
                xi = np.arange(-self.FineTuneNum, (self.FineTuneNum + 1), 1) / self.FineTuneNum * (
                            self.FineTuneDis / dl) + c_dat[ix][0]
                yi = a * xi + b
                imtmp = f(xi, yi)
                y = np.array([imtmp[ii, ii] for ii in range(self.FineTuneNum*2)])
                dy = y[1:] - y[:-1]
                if flip_flg:
                    kk = np.argmax(dy)
                else:
                    kk = np.argmin(dy)
            x_ = (xi[kk] + xi[kk+1])/2
            y_ = (yi[kk] + yi[kk+1])/2
            ret_xy.append([x_, y_])
        ret_xy.append(c_dat[-1])
        po = np.array(ret_xy)
        return po

    def fine_tune(self, img_np, flip_flg):
        self.point[self.fr_num][1:-1] = self.fine_tune_base(img_np, flip_flg)[1:-1]


class VtParts(VtPartsBase):
    UnEdit_C = 'green'
    Edit_C = 'cyan'
    Active_C = 'GreenYellow'

    def __init__(self, tkcanvas, name='tongue', dat=None, EditFlg=False, mat_affine=None, win=None):
        super().__init__(name, dat)
        self.mat_affine = mat_affine # 保存用
        self.dr_point = None    # キャンパス上の位置　対処のフレームのみ np.array
        self.dr_point_save = []
        self.EditFlg = EditFlg  # 修正対象か？
        self.vt_point = []         # 作業用のVtPointの配列
        self.tkcanvas = tkcanvas
        self.win=win

        self.sel_st_p = 1000   # 作業用
        self.sel_ed_p = -1   # 作業用

        self.tk_line_id = None

    def reset_para(self):
        self.sel_st_p = 1000
        self.sel_ed_p = -1
        self.dr_point_save = []
        self.dr_point = None
        for vtp in self.vt_point:
            vtp.reset_para()

    def backup(self):
        self.dr_point_save.append(np.array(self.dr_point))

    def get_drpoint(self):
        dr_p = (self.mat_affine[:2, :2] @ self.point[self.fr_num].T).T + self.mat_affine[:2, -1]
        return dr_p

    def dr_to_point(self):
        mat_inv = np.linalg.inv(self.mat_affine)
        self.point[self.fr_num] = (mat_inv[:2, :2] @ self.dr_point.T).T + mat_inv[:2, -1]

    def delete_item(self):
        if self.tk_line_id:
            self.tkcanvas.delete(self.tk_line_id)
        for vtp in self.vt_point:
            vtp.delete_item()

    def click_func(self, ev):
        self.win.write_event_value("LineSelect", self.name)

    def draw(self, fr_num):
        self.fr_num = fr_num
        self.dr_point = self.get_drpoint()
        self.backup()
        pline = self.dr_point.flatten().tolist()

        if self.tk_line_id:
            self.tkcanvas.delete(self.tk_line_id)
        for vtp in self.vt_point:
            vtp.delete_item()
        if self.EditFlg:
            self.tk_line_id = self.tkcanvas.create_line(*pline, fill=self.Edit_C, activefill=self.Active_C)
            self.vt_point = []
            for ix,pp in enumerate(self.dr_point):
                self.vt_point.append(VtPoint(ix, pp, self.select_func, self.move_func, self.tkcanvas))
        else:
            self.tk_line_id = self.tkcanvas.create_line(*pline, fill=self.UnEdit_C, activefill=self.Active_C)

        self.tkcanvas.tag_bind(self.tk_line_id, '<Button-1>', self.click_func)

    def redraw(self):
        pline = self.dr_point.flatten().tolist()
        self.tkcanvas.delete(self.tk_line_id)
        if self.EditFlg:
            self.tk_line_id = self.tkcanvas.create_line(*pline, fill=self.Edit_C, activefill=self.Active_C)
            for vt_p in self.vt_point:
                vt_p.redraw()
        else:
            self.tk_line_id = self.tkcanvas.create_line(*pline, fill=self.UnEdit_C, activefill=self.Active_C)
            for vt_p in self.vt_point:
                vt_p.delete_item()
            self.vt_point = []

        self.tkcanvas.tag_bind(self.tk_line_id, '<Button-1>', self.click_func)

    def move_func(self, mode, ix, ev, data=None):
        if mode == 'st':
            self.backup()
        elif mode == 'mv':
            for ii in range(self.sel_st_p, ix):
                dxy = data * (ii - self.sel_st_p +1) / (ix - self.sel_st_p +1)
                self.vt_point[ii].move(dxy)
            for ii in range(ix+1, self.sel_ed_p+1):
                dxy = data * (self.sel_ed_p - ii +1) / (self.sel_ed_p- ix +1)
                self.vt_point[ii].move(dxy)
            self.vt_point[ix].move(data)

        elif mode == 'rd':
            self.dr_point = np.array([p.dr_posi for p in self.vt_point])
            self.redraw()

    def select_func(self, mode, ix):
        if mode == 'add':
            if ix < self.sel_st_p:
                self.sel_st_p = ix
            if ix > self.sel_ed_p:
                self.sel_ed_p = ix
            for ii in range(self.sel_st_p, self.sel_ed_p+1):
                self.vt_point[ii].select()
        elif mode == 'set_s':
            self.del_select_all()
            self.sel_st_p = ix - 1 if ix > 0 else 0

            self.sel_ed_p = ix + 1 if ix < len(self.vt_point)-1 else len(self.vt_point)-1
            for ii in range(self.sel_st_p, self.sel_ed_p+1):
                self.vt_point[ii].select()
        elif mode == 'del':
            self.del_select_all()

    def del_select_all(self):
        for ii in range(self.sel_st_p, self.sel_ed_p+1):
            self.vt_point[ii].unselect()
        self.sel_st_p = 1000
        self.sel_ed_p = -1

    def select_all(self):
        self.sel_st_p = 0
        self.sel_ed_p = len(self.vt_point)-1
        for ii in range(self.sel_st_p, self.sel_ed_p+1):
            self.vt_point[ii].select()

    def reset_dr_posi(self):
        for ix, vp in enumerate(self.vt_point):
            vp.dr_posi = self.dr_point[ix]

    def undo(self):
        if len(self.dr_point_save) > 0:
            self.dr_point = self.dr_point_save.pop(-1)
            self.reset_dr_posi()

    def smoothing(self):
        self.dr_point = smoothing(self.dr_point, self.sel_st_p, self.sel_ed_p)
        self.reset_dr_posi()

    def resample(self):
        self.dr_point = resample(self.dr_point, self.resample_num)
        self.reset_dr_posi()

    def fine_tune(self, img_np, flip_flg):
        po = self.fine_tune_base(img_np, flip_flg)

        for ix in range(len(po)):
            if not self.vt_point[ix].selected:
                po[ix] = self.point[self.fr_num][ix]
        self.point[self.fr_num][1:-1] = po[1:-1]
        self.dr_point = self.get_drpoint()
        self.reset_dr_posi()


class VTShapeBase:
    AllList = ['tongue', 'uplips', 'lowerlips', 'palate', 'uppharynx', 'lowpharynx', 'larynx']
    def __init__(self, part='tongue'):
        self.part_name = part
        self.PNum = {'tongue':40,  # % 舌の点数-------------------変更可能----
                     'uplips':15,  # % 上唇の点数-----------------変更可能----
                     'lowerlips':25,  # % 下唇の点数-----------------変更可能----
                     'palate':30,  # % 軟口蓋の点数----------------変更可能----
                     'uppharynx':14,  # % 咽頭の点数------------------変更可能----
                     'lowpharynx':14,  # % 咽頭の点数------------------変更可能----
                     'larynx':30  # % 喉頭蓋の点数-----------------変更可能----
                     }
        self.parts = {}

    def __len__(self):
        if self.part_name in self.parts:
            return len(self.parts[self.part_name])
        else:
            return 0

    def read_dat(self, fname):
        dat = np.loadtxt(fname, delimiter=',')
        if dat.shape[1] > 300:
            st = 0
            for name in self.AllList:
                ed = st + self.PNum[name] * 2
                self.parts[name] = VtPartsBase(name, dat[:, st:ed])
                st = ed
        else:
            self.parts[self.part_name] = VtPartsBase(self.part_name, dat)

    def save_dat(self, fname):
        dat = self.parts[self.AllList[0]].point
        ll = len(dat)
        dat = dat.reshape([ll, -1])
        for pname in self.AllList[1:]:
            dat = np.concatenate([dat, self.parts[pname].point.reshape([ll, -1])], 1)
        np.savetxt(fname, dat, fmt='%.2f', delimiter=',')



class VTShape(VTShapeBase):
    def __init__(self, canvas, cfg, part='tongue', fr_num=0, win=None):
        super().__init__(part)
        self.cfg = cfg
        self.win = win
        self.canvas = canvas
        self.tkcanvas = canvas.get_tkwidget()
        self.fr_num = fr_num

        self.init_class_values(cfg)

        self.read_dat()

        self.tkcanvas.bind('<Button-3>', self.del_select_all)

    def init_class_values(self, cfg):
        VtPartsBase.FineTuneDis=cfg.finetune_distance

    def reset_para(self):
        for pname in self.parts.keys():
            self.parts[pname].reset_para()

    def del_select_all(self, ev):
        self.parts[self.part_name].del_select_all()

    def select_all(self):
        self.parts[self.part_name].select_all()

    def read_dat(self, fn=None):
        if fn:
            fname = fn
        else:
            fname = self.cfg.file.format(dir=self.cfg.dir, spk=self.cfg.spk, fname=self.cfg.fname)
        dat = np.loadtxt(fname, delimiter=',')
        if dat.shape[1] > 300:
            st = 0
            for name in self.AllList:
                ed = st + self.PNum[name]*2
                self.parts[name] = VtParts(self.tkcanvas, name, dat[:, st:ed], self.part_name==name, mat_affine=self.canvas.mat_affine, win=self.win)
                st = ed
        else:
            self.parts[self.part_name] = VtParts(self.tkcanvas, self.part_name, dat, True, win=self.win)

    def save_dat(self, fname):
        dat = self.parts[self.AllList[0]].point
        ll = len(dat)
        dat = dat.reshape([ll, -1])
        for pname in self.AllList[1:]:
            dat = np.concatenate([dat, self.parts[pname].point.reshape([ll, -1])], 1)
        np.savetxt(fname, dat, fmt='%.2f', delimiter=',')

    def delete_item(self):
        for ptn in self.parts.keys():
            self.parts[ptn].delete_item()

    def draw(self):
        for pname in self.AllList:
            self.parts[pname].draw(self.fr_num)

    def undo(self):
        self.parts[self.part_name].undo()
        self.parts[self.part_name].redraw()

    def smoothing(self):
        self.parts[self.part_name].smoothing()
        self.parts[self.part_name].redraw()

    def resample(self):
        self.parts[self.part_name].resample()
        self.parts[self.part_name].redraw()

    def dr_to_point(self):
        self.parts[self.part_name].dr_to_point()

    def fine_tune(self, img_np):
        flip_flg = True if self.part_name in ['uplips', 'lowerlips', 'uppharynx', 'lowpharynx'] else False
        self.parts[self.part_name].fine_tune(img_np, flip_flg)
        self.parts[self.part_name].redraw()

    def change_parts(self, pname):
        if pname == self.part_name:
            return
        self.parts[self.part_name].EditFlg = False
        self.parts[self.part_name].redraw()
        self.part_name = pname
        self.parts[self.part_name].EditFlg = True
        self.parts[self.part_name].draw(self.fr_num)

    def backup(self):
        self.parts[self.part_name].backup()