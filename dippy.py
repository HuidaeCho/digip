#!/usr/bin/env python3
################################################################################
# Name:    DIPPy: A Digital Image Processing Python module
# Purpose: This module provides various digital image processing functions.
# Author:  Huidae Cho
# Since:   February 2, 2018
#
# Copyright (C) 2019, Huidae Cho <https://idea.isnew.info>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import math

def show(img, L=256, scale=False, stretch=False):
    '''
    Shows a grayscale image

    img:        Grayscale image
    L:          Gray levels (default 256)
    scale:      True for scaled images, False for the original size (default)
    stretch:    True for stretched grayscale,
                False for 0 to L-1 gray levels (default)
    '''
    if stretch:
        vmin = np.min(img)
        vmax = np.max(img)
    else:
        vmin = 0.
        vmax = L-1.
    if scale:
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        dpi = plt.rcParams['figure.dpi']
        plt.figure(figsize=(img.shape[0]/dpi, img.shape[1]/dpi))
        plt.figimage(img, 0, 0, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()

def compare(img1, img2, L=256, scale=False, stretch=False):
    '''
    Compares two grayscale images

    img1:       Image 1
    img2:       Image 2
    L:          Gray levels (default 256)
    scale:      True for scaled images, False for the original size (default)
    stretch:    True for stretched grayscale,
                False for 0 to L-1 gray levels (default)
    '''
    if stretch:
        vmin1 = np.min(img1)
        vmax1 = np.max(img1)
        vmin2 = np.min(img2)
        vmax2 = np.max(img2)
    else:
        vmin1 = vmin2 = 0.
        vmax1 = vmax2 = L-1.
    if scale:
        fig, ax = plt.subplots(1, 2)
        ax[0].set_title('img1')
        ax[0].set_axis_off()
        ax[0].imshow(img1, cmap="gray", vmin=vmin1, vmax=vmax1)
        ax[1].set_title('img2')
        ax[1].set_axis_off()
        ax[1].imshow(img2, cmap="gray", vmin=vmin2, vmax=vmax2)
    else:
        dpi = plt.rcParams['figure.dpi']
        plt.figure(figsize=((img1.shape[0]+img2.shape[0])/dpi,
                            max(img1.shape[1], img2.shape[1])/dpi))
        figx = 0
        figy = max(img2.shape[1]-img1.shape[1], 0)
        plt.figimage(img1, figx, figy, cmap='gray', vmin=vmin1, vmax=vmax1)
        figx += img1.shape[0]
        figy = max(img1.shape[1]-img2.shape[1], 0)
        plt.figimage(img2, figx, figy, cmap='gray', vmin=vmin2, vmax=vmax2)
    plt.show()

def convert_to_grayscale(img):
    '''
    Convert a color image to grayscale (256 levels)

    img:    Input image
    Returns grayscale image
    '''
    gray_img = np.dot(img, [.299, .587, .114])
    return gray_img

def clip(img, start, end):
    '''
    Clip an image

    img:    Input image
    start:  Start (row, column)
    end:    End (row, column)
    Returns clipped image
    '''
    clipped = img[start[0]:end[0], start[1]:end[1], :]
    return clipped

def row_column_delete(img):
    '''
    Row-column delete

    img:    Input image
    Returns shrinked image
    '''
    new_shape = [int(x / 2) for x in img.shape]
    new_img = np.empty(new_shape)
    for i in range(0, new_shape[0]):
        for j in range(0, new_shape[1]):
            new_img[i,j] = img[2*i,2*j]
    return new_img

def nearest_neighbor_interpolate(img, scale):
    '''
    Nearest neighbor interpolation

    img:    Input image
    scale:  Scalar scale
    Returns interpolated image
    '''
    new_shape = [int(scale * x) for x in img.shape]
    new_img = np.empty(new_shape)
    for i in range(0, new_shape[0]):
        for j in range(0, new_shape[1]):
            row = int(i / scale)
            col = int(j / scale)
            new_img[i,j] = img[row,col]
    return new_img

def bilinear_interpolate(img, scale):
    '''
    Bilinear interpolation

    img:    Input image
    scale:  Scalar scale
    Returns interpolated image
    '''
    def get_corner_neighbor(img, x, y):
        return img[int(y),int(x)]

    def get_vertical_border_neighbor(img, x, y):
        x1 = x2 = int(x)
        y1 = int(y)
        if y-y1 < .5:
            y2 = y1 - 1
        else:
            y2 = y1 + 1

        f1 = int(img[y1,x1])
        f2 = int(img[y2,x2])

        return f1 + (f2-f1) / (y2-y1) * (y-y1)

    def get_horizontal_border_neighbor(img, x, y):
        x1 = int(x)
        if x-x1 < .5:
            x2 = x1 - 1
        else:
            x2 = x1 + 1
        y1 = y2 = int(y)

        f1 = int(img[y1,x1])
        f2 = int(img[y2,x2])

        return f1 + (f2-f1) / (x2-x1) * (x-x1)

    def get_inner_neighbor(img, x, y):
        x1 = int(x)
        if x-x1 < .5:
            x2 = x1 - 1
        else:
            x2 = x1 + 1
        y1 = int(y)
        if y-y1 < .5:
            y2 = y1 - 1
        else:
            y2 = y1 + 1

        f11 = int(img[y1,x1])
        f12 = int(img[y1,x2])
        f21 = int(img[y2,x1])
        f22 = int(img[y2,x2])

        a1 = (f21-f11) / (x2-x1)
        b1 = (f11*x2-f21*x1) / (x2-x1)
        a2 = (f22-f12) / (x2-x1)
        b2 = (f12*x2-f22*x1) / (x2-x1)
        a = (a1*y2-a2*y1) / (y2-y1)
        b = (b2-b1) / (y2-y1)
        c = (a2-a1) / (y2-y1)
        d = (b1*y2-b2*y1) / (y2-y1)

        return a*x+b*y+c*x*y+d

    new_shape = [int(scale * x) for x in img.shape]
    new_img = np.empty(new_shape)
    for i in range(0, new_shape[0]):
        for j in range(0, new_shape[1]):
            x = j / float(scale)
            y = i / float(scale)
            if ((x <= .5 and (y <= .5 or y >= img.shape[1]-1.5)) or
                (x >= img.shape[0]-1.5 and (y <= .5 or y >= img.shape[1]-1.5))):
                # corner: one neighbor
                new_img[i,j] = get_corner_neighbor(img, x, y)
            elif x <= .5 or x >= img.shape[0]-1.5:
                # vertical borders: two neighbors
                new_img[i,j] = get_vertical_border_neighbor(img, x, y)
            elif y <= .5 or y >= img.shape[1]-1.5:
                # horizontal borders: two neighbors
                new_img[i,j] = get_horizontal_border_neighbor(img, x, y)
            else:
                # inner pixels: four neighbors
                new_img[i,j] = get_inner_neighbor(img, x, y)

    return new_img

def grayscale_transform(img, new_L, L=256):
    '''
    Grayscale transform

    new_L:  New gray levels
    L:      Original gray levels (default 256)
    Returns grayscaled image
    '''
    r = img
    s = r.copy()
    for c in range(0, new_L):
        s[(s >= c*L/new_L) & (s < (c+1)*L/new_L)] = c
    return s

def negative_transform(img, L=256):
    '''
    Negative transform

    img:    Input image
    L:      Gray levels (default 256)
    Returns transformed image
    '''
    r = img
    s = L - 1. - r
    return s

def linear_transform(img, cp1, cp2, L=256):
    '''
    Linear transform using two control points

    img:    Input image
    cp1:    Control point 1 (r1, s1)
    cp2:    Control point 2 (r2, s2)
    L:      Gray levels (default 256)
    Returns transformed image
    '''
    r = img
    r1 = cp1[0]
    s1 = cp1[1]
    r2 = cp2[0]
    s2 = cp2[1]
    if r1 == r2:
        s = np.where(r < r1, s1 / r1 * r,
                s2 + (L-1. - s2) / (L-1. - r2) * (r - r2))
    else:
        s = np.where(r < r1, s1 / r1 * r,
                np.where(r < r2, s1 + (s2 - s1) / (r2 - r1) * (r - r1),
                    s2 + (L-1. - s2) / (L-1. - r2) * (r - r2)))
    return s

def log_transform(img, L=256):
    '''
    Log transform

    img:    Input image
    L:      Gray levels (default 256)
    Returns transformed image
    '''
    r = img
    c = (L - 1.) / math.log10(1. + np.max(r))
    s = c * np.log10(1. + r)
    return s

def inverse_log_transform(img, L=256):
    '''
    Inverse log transform

    img:    Input image
    L:      Gray levels (default 256)
    Returns transformed image
    '''
    r = img
    c = (L - 1.) / 10.**np.max(r)
    s = c * (10.**r - 1.)
    return s

def power_transform(img, gamma, L=256):
    '''
    Power transform

    img:    Input image
    gamma:  gamma
    L:      Gray levels (default 256)
    Returns transformed image
    '''
    r = img
    c = (L - 1.) / np.max(r)**gamma
    s = c * r**gamma
    return s