#!/usr/bin/env python3
################################################################################
# Name:    Digip: A Digital Image Processing Python module
# Purpose: This module provides various digital image processing functions.
# Author:  Huidae Cho
# Since:   February 2, 2019
#
# Copyright (C) 2019 Huidae Cho <https://faculty.ung.edu/hcho/>
#                    Institute for Environmental and Spatial Analysis (IESA)
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
import random
import warnings

def read_image(filename):
    '''
    Read an image
    filename:   Filename
    '''
    img = plt.imread(filename)
    return img

def show_image(img, L=256, scale=False, stretch=False):
    '''
    Show an image
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

def compare_images(img1, img2, L=256, scale=False, stretch=False):
    '''
    Compare two images
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

def histogram(img, L=256, prob=False):
    '''
    Plot the histogram of a grayscale image
    img:    Input image
    L:      Gray levels (default 256)
    prob:   True for probability density, False for counts (default)
    '''
    levels = range(L)
    hist = [sum(sum(img==l)) for l in levels]
    if prob:
        hist /= sum(hist)
    plt.xlim(0., L-1.)
    plt.vlines(levels, ymin=0, ymax=hist)
    plt.show()

def convert_to_grayscale(img):
    '''
    Convert a color image to grayscale (256 levels)
    img:    Input image
    '''
    gray_img = np.dot(img, [.299, .587, .114])
    return gray_img

def clip(img, start, end):
    '''
    Clip an image
    img:    Input image
    start:  Start (row, column)
    end:    End (row, column)
    '''
    clipped = img[start[0]:end[0], start[1]:end[1], :]
    return clipped

def row_column_delete(img):
    '''
    Row-column delete
    img:    Input image
    '''
    new_shape = [int(x / 2) for x in img.shape]
    new_img = np.empty(new_shape)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            new_img[i,j] = img[2*i,2*j]
    return new_img

def nearest_neighbor_interpolate(img, scale):
    '''
    Nearest neighbor interpolate
    img:    Input image
    scale:  Scalar scale
    '''
    new_shape = [int(scale * x) for x in img.shape]
    new_img = np.empty(new_shape)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            row = int(i / scale)
            col = int(j / scale)
            new_img[i,j] = img[row,col]
    return new_img

def bilinear_interpolate(img, scale):
    '''
    Bilinear interpolate
    img:    Input image
    scale:  Scalar scale
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
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
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

def normalize_gray_levels(img, grange=(0, 255)):
    '''
    Normalize gray levels to [0, 1]
    img:    Input image
    grange: Gray-level range (min, max) (default 0, 255)
    '''
    r = img.astype(float)
    if grange == None:
        grange = (np.min(r), np.max(r))
    s = (r - grange[0]) / (grange[1] - grange[0])
    return s

def rescale_gray_levels(img, grange=(0, 1), L=256):
    '''
    Rescale gray levels to [0, L-1]
    img:    Input image
    L:      Gray levels (default 256)
    '''
    r = img.astype(float)
    if grange == None:
        grange = (np.min(r), np.max(r))
    s = ((r - grange[0]) / (grange[1] - grange[0]) * (L - 1.)).astype(int)
    s = np.clip(s, 0, L-1.)
    return s

def grayscale_transform(img, new_L, L=256):
    '''
    Grayscale transform
    img:    Input image
    new_L:  New gray levels
    L:      Original gray levels (default 256)
    '''
    r = img
    s = (r * (new_L-1.) / (L-1.)).astype(int)
    return s

def negative_transform(img, L=256):
    '''
    Negative transform
    img:    Input image
    L:      Gray levels (default 256)
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
    '''
    r = img
    c = (L - 1.) / np.max(r)**gamma
    s = c * r**gamma
    return s

def gray_level_slice(img, grange, new_gray, binary=False):
    '''
    Gray-level slice
    img:        Input image
    grange:     Gray-level range (rmin, rmax)
    new_gray:   New gray level
    binary:     True for 0 for outside grange, False for identity (default)
    '''
    r = img
    rmin = grange[0]
    rmax = grange[1]
    if binary:
        s = np.where(np.logical_and(r >= rmin, r <= rmax),
                new_gray, 0).astype(float)
    else:
        s = np.where(np.logical_and(r >= rmin, r <= rmax),
                new_gray, r).astype(float)
    return s

def bit_plane_slice(img, bit_plane):
    '''
    Bit-plane slice
    img:        Input image
    bit_plane:  Bit plane starting with 0
    '''
    r = img
    s = np.bitwise_and(r.astype(int), 1<<bit_plane).astype(float)
    return s

def histogram_equalize(img, L=256):
    '''
    Histogram equalize
    img:    Input image
    L:      Gray levels (default 256)
    '''
    r = img
    sumrk = [sum(sum(r==j))/r.size for j in range(L)]
    sk = {}
    for k in range(L):
        sk[k] = 0
        for j in range(k+1):
            sk[k] += sumrk[j]
    skmax = np.max(list(sk.values()))
    s = np.ndarray(r.shape)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            s[i,j] = int(sk[s[i,j]]/skmax*(L-1.))
    return s

def bitwise_not(img):
    '''
    Bitwise not
    img:   Input image
    '''
    g = np.bitwise_not(img.astype(int)).astype(float)
    return g

def bitwise_and(img1, img2):
    '''
    Bitwise and
    img1:   Input image 1
    img2:   Input image 2
    '''
    g = np.bitwise_and(img1.astype(int), img2.astype(int)).astype(float)
    return g

def bitwise_or(img1, img2):
    '''
    Bitwise or
    img1:   Input image 1
    img2:   Input image 2
    '''
    g = np.bitwise_or(img1.astype(int), img2.astype(int)).astype(float)
    return g

def bitwise_xor(img1, img2):
    '''
    Bitwise xor
    img1:   Input image 1
    img2:   Input image 2
    '''
    g = np.bitwise_xor(img1.astype(int), img2.astype(int)).astype(float)
    return g

def add(imgs):
    '''
    Add
    imgs:   Array of input images
    '''
    K = len(imgs)
    g = imgs[0].copy()
    for i in range(1, K):
        g += imgs[i]
    return g

def subtract(img1, img2):
    '''
    Subtract
    img1:   Input image 1
    img2:   Input image 2
    '''
    g = img1 - img2
    return g

def add_noise(img, prob, max):
    '''
    Add noise
    img:    Input image
    prob:   Probability of noise
    max:    Maximum noise
    '''
    g = img.copy()
    for i in range(g.shape[0]):
        for j in range(g.shape[1]):
            if random.random() < prob:
                g[i,j] += random.randint(-max, max)
    return g

def average(imgs):
    '''
    Average images
    imgs:   Array of input images
    '''
    g = add(imgs) / len(imgs)
    return g

def find_neighborhood(img, current, size):
    '''
    Find neighborhood
    img:        Input image
    current:    Current pixel (row, column)
    size:       Neighborhood size (rows, columns)
    '''
    half_row_size = int((size[0]-1)/2)
    half_col_size = int((size[1]-1)/2)
    row_min = max(current[0]-half_row_size, 0)
    row_max = min(current[0]+half_row_size, img.shape[0]-1)
    col_min = max(current[1]-half_col_size, 0)
    col_max = min(current[1]+half_col_size, img.shape[1]-1)
    S = img[row_min:row_max+1, col_min:col_max+1]
    return S

def local_statistics(img, size, stats):
    '''
    Local statistics
    img:    Input image
    size:   Neighborhood size (rows, columns)
    stats:  Statistics to calculate
            Bitwise-or value:
             1 mean
             2 standard deviation
             4 median
             8 minimum
            16 maximum
    '''
    half_row_size = int((size[0]-1)/2)
    half_col_size = int((size[1]-1)/2)
    if stats & 1:
        mean = np.ndarray(img.shape)
    else:
        mean = None
    if stats & 2:
        std = np.ndarray(img.shape)
    else:
        std = None
    if stats & 4:
        median = np.ndarray(img.shape)
    else:
        median = None
    if stats & 8:
        minimum = np.ndarray(img.shape)
    else:
        minimum = None
    if stats & 16:
        maximum = np.ndarray(img.shape)
    else:
        maximum = None
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # find neighborhood
            row_min = max(row-half_row_size, 0)
            row_max = min(row+half_row_size, img.shape[0]-1)
            col_min = max(col-half_col_size, 0)
            col_max = min(col+half_col_size, img.shape[1]-1)
            S = img[row_min:row_max+1, col_min:col_max+1]
            # calculate statistics
            if stats & 1:
                mean[row,col] = np.mean(S)
            if stats & 2:
                std[row,col] = np.std(S)
            if stats & 4:
                median[row,col] = np.median(S)
            if stats & 8:
                minimum[row,col] = np.min(S)
            if stats & 16:
                maximum[row,col] = np.max(S)
    return mean, std, median, minimum, maximum

def local_enhance(img, local_mean, local_std, mult, k):
    '''
    Local enhance
    img:        Input image
    local_mean: Local mean image
    local_std:  Local standard deviation imagge
    mult:       Gray-level multiplier
    k:          Criteria parameters (k0, k1, k2)
    '''
    if mult == 1:
        # identity
        g = img.copy()
    else:
        mean = np.mean(img)
        std = np.std(img)
        if mult < 1:
            # enhance bright pixels
            g = np.where(np.logical_and(local_mean >= k[0]*mean,
                    np.logical_and(local_std >= k[1]*std,
                        local_std <= k[2]*std)), mult*img, img)
        else:
            # enhance dark pixels
            g = np.where(np.logical_and(local_mean <= k[0]*mean,
                    np.logical_and(local_std >= k[1]*std,
                        local_std <= k[2]*std)), mult*img, img)
    return g

def convolute(img, mask, average=False):
    '''
    Convolute
    img:        Input image
    mask:       Mask
    average:    True for weighted average, False for convolution (default)
    '''
    half_row_size = int((mask.shape[0]-1)/2)
    half_col_size = int((mask.shape[1]-1)/2)
    g = img.astype(float)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            # find neighborhood
            row_min = max(row-half_row_size, 0)
            row_max = min(row+half_row_size, img.shape[0]-1)
            col_min = max(col-half_col_size, 0)
            col_max = min(col+half_col_size, img.shape[1]-1)
            S = img[row_min:row_max+1, col_min:col_max+1]

            mask_row_min = 0
            if row - half_row_size < 0:
                mask_row_min = half_row_size - row
            mask_row_max = mask.shape[0] - 1
            if row + half_row_size >= img.shape[0]:
                mask_row_max = mask_row_max - (row + half_row_size - img.shape[0])
            mask_col_min = 0
            if col - half_col_size < 0:
                mask_col_min = half_col_size - col
            mask_col_max = mask.shape[1] - 1
            if col + half_col_size >= img.shape[1]:
                mask_col_max = mask_col_max - (col + half_col_size - img.shape[1])
            if mask_row_max - mask_row_min != row_max - row_min:
                if row - half_row_size < 0:
                    while mask_row_max - mask_row_min != row_max - row_min:
                        mask_row_min += 1
                else:
                    while mask_row_max - mask_row_min != row_max - row_min:
                        mask_row_max -= 1
            if mask_col_max - mask_col_min != col_max - col_min:
                if col - half_col_size < 0:
                    while mask_col_max - mask_col_min != col_max - col_min:
                        mask_col_min += 1
                else:
                    while mask_col_max - mask_col_min != col_max - col_min:
                        mask_col_max -= 1
            w = mask[mask_row_min:mask_row_max+1,
                     mask_col_min:mask_col_max+1]

            if average:
                g[row,col] = np.sum(S*w)/np.sum(w)
            else:
                g[row,col] = np.sum(S*w)
    return g

def weighted_average(img, weights):
    '''
    Weighted average
    img:        Input image
    weights:    Weights
    '''
    g = convolute(img, weights, True);
    return g

def first_derivative(img):
    '''
    First derivative using the Sobel filter
    img:    Input image
    '''
    mask_x = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]])
    mask_y = np.array([[-1,  0,  1],
                       [-2,  0,  2],
                       [-1,  0,  1]])
    g = np.abs(convolute(img, mask_x))+np.abs(convolute(img, mask_y))
    return g

def second_derivative(img, diag=True):
    '''
    Second derivative using the Laplacian filter
    img:    Input image
    diag:   True for diagonal directions (default),
            False for no diagonal directions
    '''
    if diag:
        mask = np.array([[1,  1, 1],
                         [1, -8, 1],
                         [1,  1, 1]])
    else:
        mask = np.array([[0,  1, 0],
                         [1, -4, 1],
                         [0,  1, 0]])
    g = convolute(img, mask)
    return g

def sharpen(img, diag=True, L=256):
    '''
    Laplacian sharpen
    img:    Input image
    diag:   True for diagonal directions (default),
            False for no diagonal directions
    L:      Gray levels (default 256)
    '''
    if diag:
        mult = 8
    else:
        mult = 4
    grange = tuple(mult*x for x in (-L+1., L-1.))

    lap = second_derivative(img, diag)
    g = img - rescale_gray_levels(lap, grange, L)
    g = rescale_gray_levels(g, (np.min(g), np.max(g)), L)
    return g

def high_boost_filter(img, A=1, diag=True):
    '''
    High-boost filter using the Laplacian
    img:    Input image
    A:      High-boost filter parameter
    diag:   True for diagonal directions (default),
            False for no diagonal directions
    '''
    if diag:
        mask = np.array([[-1,  -1, -1],
                         [-1, A+8, -1],
                         [-1,  -1, -1]])
    else:
        mask = np.array([[ 0,  -1,  0],
                         [-1, A+4, -1],
                         [ 0,  -1,  0]])
    g = convolute(img, mask)
    return g

def dft_1d(f):
    '''
    1-dimensional discrete Fourier transform
    f:  Original image
    '''
    M = len(f)
    F = np.zeros(M, dtype=complex)
    for u in range(M):
        for x in range(M):
            F[u] += f[x]*np.exp(-2j*np.pi*u*x/M)/M
    return F

def idft_1d(F):
    '''
    1-dimensional inverse discrete Fourier transform
    F:  Discrete Fourier transform
    '''
    M = len(F)
    # this conjugate algorithm is simpler, but it may be slower because two
    # conjugate operations are required "additionally"; let's keep the full
    # version just in case
    return np.conj(M*dft_1d(np.conj(F)))
    #f = np.zeros(M, dtype=complex)
    #for x in range(M):
    #    for u in range(M):
    #        f[x] += F[u]*np.exp(2j*np.pi*u*x/M)
    #return f

def fft_1d(f):
    '''
    1-dimensional fast Fourier transform
    f:  Original image
    '''
    M = len(f)
    K = int(M/2)
    feven = f[::2]
    fodd = f[1::2]
    if K == 1:
        Feven = feven[0]/2
        Fodd = fodd[0]/2
        F = np.array((Feven+Fodd, Feven-Fodd))
    else:
        Feven = fft_1d(feven)/2
        Fodd = fft_1d(fodd)/2*np.exp(-2j*np.pi*np.arange(K)/(2*K))
        F = np.concatenate((Feven+Fodd, Feven-Fodd))
    return F

def ifft_1d(F):
    '''
    1-dimensional inverse fast Fourier transform
    F:  Discrete Fourier transform
    '''
    M = len(F)
    # this conjugate algorithm is simpler, but it may be slower because two
    # conjugate operations are required "additionally"; let's keep the full
    # version just in case
    return np.conj(M*fft_1d(np.conj(F)))
    #K = M/2
    #Feven = F[::2]
    #Fodd = F[1::2]
    #if K == 1:
    #    feven = Feven[0]/2
    #    fodd = Fodd[0]/2
    #    f = np.array((feven+fodd, feven-fodd))
    #else:
    #    feven = ifft_1d(Feven)/2
    #    fodd = ifft_1d(Fodd)/2*np.exp(2j*np.pi*np.arange(K)/(2*K))
    #    f = np.concatenate((feven+fodd, feven-fodd))
    #return f

def fft(f):
    '''
    2-dimensional fast Fourier transform
    f:  Original image
    '''
    M, N = f.shape
    Fx = np.zeros(f.shape, dtype=complex)
    for x in range(M):
        Fx[x] = fft_1d(f[x])
    F = np.zeros(f.shape, dtype=complex)
    for v in range(N):
        F[...,v] = fft_1d(Fx[...,v])
    return F

def ifft(F):
    '''
    2-dimensional inverse fast Fourier transform
    F:  Discrete Fourier transform
    '''
    M, N = F.shape
    # this conjugate algorithm is simpler, but it may be slower because two
    # conjugate operations are required "additionally"; let's keep the full
    # version just in case
    return np.conj(M*N*fft(np.conj(F)))
    #fx = np.zeros(F.shape, dtype=complex)
    #for x in range(M):
    #    fx[x] = ifft_1d(F[x])
    #f = np.zeros(F.shape, dtype=complex)
    #for v in range(N):
    #    f[...,v] = ifft_1d(fx[...,v])
    #return f

def pad(x):
    '''
    Pad zeros to x so x.shape becomes a power of two
    x:  2-dimensional array
    '''
    m, n = x.shape
    M = 2**math.ceil(math.log2(m))
    N = 2**math.ceil(math.log2(n))
    if m != M or n != N:
        M -= m
        N -= n
        t = int(M/2)
        b = M-t
        l = int(N/2)
        r = N-l
        x = np.pad(x, ((t,b), (l,r)), 'constant')
    return x

def shift(x):
    '''
    Shift x by (-1)**(row+column)
    x:  2-dimensional array
    '''
    M, N = x.shape
    y = np.ndarray(x.shape)
    for r in range(M):
        for c in range(N):
            y[r,c] = x[r,c]*(-1)**(r+c)
    return y

def spectrum(F):
    '''
    Calculate the Fourier spectrum
    F:  Discrete Fourier transform
    '''
    return abs(F)

def phase_angle(F):
    '''
    Calculate the Fourier phase angle
    F:  Discrete Fourier transform
    '''
    return np.arctan2(F.imag, F.real)

def power_spectrum(F):
    '''
    Calculate the Fourier power spectrum
    F:  Discrete Fourier transform
    '''
    return spectrum(F)**2

def ideal_lowpass_filter(size, D0):
    '''
    Ideal low-pass filter
    size:   M, N tuple
    D0:     Cut-off frequency
    '''
    H = np.ndarray(size)
    M = size[0]
    N = size[1]
    for u in range(M):
        for v in range(N):
            D = math.sqrt((u-M/2)**2+(v-N/2)**2)
            if D <= D0:
                print([u-M/2, v-M/2])
            H[u,v] = 1 if D <= D0 else 0
    return H

def gaussian_lowpass_filter(size, D0):
    '''
    Gaussian low-pass filter
    size:   M, N tuple
    D0:     Cut-off frequency
    '''
    H = np.ndarray(size)
    M = size[0]
    N = size[1]
    for u in range(M):
        for v in range(N):
            D = math.sqrt((u-M/2)**2+(v-N/2)**2)
            H[u,v] = np.exp(-D**2/(2*D0**2))
    return H

def gaussian_highpass_filter(size, D0):
    '''
    Gaussian high-pass filter
    size:   M, N tuple
    D0:     Cut-off frequency
    '''
    return 1 - gaussian_lowpass_filter(size, D0)

def rgb_to_cmy(R, G, B, maxRGB=255):
    '''
    Convert RGB to CMY
    R:      R
    G:      G
    B:      B
    maxRGB: Maximum RGB (255 by default)
    '''
    C = maxRGB-R;
    M = maxRGB-G;
    Y = maxRGB-B;
    return C, M, Y

def cmy_to_rgb(C, M, Y, maxCMY=255):
    '''
    Convert CMY to RGB
    C:      C
    M:      M
    Y:      Y
    maxCMY: Maximum CMY (255 by default)
    '''
    R = maxCMY-C;
    G = maxCMY-M;
    B = maxCMY-Y;
    return R, G, B

def rgb_to_hsi(R, G, B, maxRGB=255, Hnorm=False):
    '''
    Convert RGB to HSI
    R:      R
    G:      G
    B:      B
    maxRGB: Maximum RGB (255 by default)
    Hnorm:  True for normalized H
            False for H in degrees (default)
    '''
    r = R/maxRGB
    g = G/maxRGB
    b = B/maxRGB
    I = (r+g+b)/3
    with warnings.catch_warnings():
        # NaNs are expected when r=g=b
        # Python throws a warning on NaN, but ArcGIS Pro just fails
        warnings.simplefilter('ignore')
        theta = np.arccos(((r-g)+(r-b))/2/np.sqrt((r-g)**2+(r-b)*(g-b)))/np.pi*180
        H = np.where(g >= b, theta, 360-theta)
        if Hnorm:
            H /= 360
        # clip S just in case there are small negative numbers
        S = np.clip(1-np.minimum.reduce([r, g, b])/I, 0, 1)
    return H, S, I

def hsi_to_rgb(H, S, I, Hnorm=False, maxRGB=255):
    '''
    Convert HSI to RGB
    H:      H
    S:      S
    I:      I
    Hnorm:  True for normalized H
            False for H in degrees (default)
    maxRGB: Maximum RGB (255 by default)
    '''
    if Hnorm:
        H *= 360
    with warnings.catch_warnings():
        # NaNs are expected when r=g=b
        # Python throws a warning on NaN, but ArcGIS Pro just fails
        warnings.simplefilter('ignore')
        case = np.where(np.isnan(H), 0, np.where(H < 120, 1, np.where(H < 240, 2, 3)))
        H = np.where(case == 0, 0, np.where(case == 1, H, np.where(case == 2, H-120, H-240)))
        c1 = np.where(case == 0, I, I*(1-S))
        c2 = np.where(case == 0, I, I*(1+S*np.cos(H/180*np.pi)/np.cos((60-H)/180*np.pi)))
        c3 = np.where(case == 0, I, 3*I-(c1+c2))
    R = np.where(case <= 1, c2, np.where(case == 2, c1, c3))*maxRGB
    G = np.where(case <= 1, c3, np.where(case == 2, c2, c1))*maxRGB
    B = np.where(case <= 1, c1, np.where(case == 2, c3, c2))*maxRGB
    return R, G, B
