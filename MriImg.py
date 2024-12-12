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

from PIL import Image, ImageTk, ImageOps
# ログの設定
logger = logging.getLogger(__name__)

class MriBase:
    def __init__(self, fname):
        self.pil_image = self.read_file(fname)

    def read_file(self, fname):
        if fname.endswith('npy'):
            npimg = np.load(fname)
            return Image.fromarray(npimg)
        else:
            return Image.open(fname)

    def get_size(self):
        return self.pil_image.width, self.pil_image.height

class Mri(MriBase):
    def __init__(self, canvas, cfg, num=0, num_base=0):
        self.canvas = canvas
        self.cfg = cfg
        self.num = num
        self.num_base = cfg.num_base

        self.read_file()

    def read_file(self, fname=None, frnum=None, spk=None):
        if fname is None:
            fname = self.cfg.fname
        if frnum is not None:
            self.num = frnum
        if spk is None:
            spk = self.cfg.spk
        num = self.num + self.num_base
        fname = self.cfg.file.format(dir=self.cfg.dir, spk=spk, fname=fname, frame_num=num, ext=self.cfg.ext)

        self.pil_image = super().read_file(fname)

    def get_image(self, mat_affine):
        # キャンバスから画像データへのアフィン変換行列を求める
        #（表示用アフィン変換行列の逆行列を求める）
        mat_inv = np.linalg.inv(mat_affine)

        # PILの画像データをアフィン変換する
        dst = self.pil_image.transform(
                    self.canvas.get_size(),  # 出力サイズ
                    Image.Transform.AFFINE,                   # アフィン変換
                    tuple(mat_inv.flatten()),       # アフィン変換行列（出力→入力への変換行列）を一次元のタプルへ変換
                    Image.Resampling.NEAREST,                  # 補間方法、ニアレストネイバー
                    #fillcolor= self.back_color
                    )

        # 表示用画像を保持
        return ImageTk.PhotoImage(image=dst)

    def draw(self):
        self.canvas.get_tkwidget().delete("all")

        self.mri_image = self.get_image(self.canvas.mat_affine)

        # 画像の描画
        self.canvas.get_tkwidget().create_image(
                0, 0,               # 画像表示位置(左上の座標)
                anchor='nw',        # アンカー、左上が原点
                image=self.mri_image    # 表示画像データ
                )

