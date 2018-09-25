#!/usr/bin/env python3
#Author: Zhu Yong-kai (yongkai_zhu@hotmail.com)
import os
import argparse

from PIL import Image
import numpy as np


def linear_interpolation(x, x1, x2, fq1, fq2):
    factor = (x2 - x1)
    factor1 = (x2 - x)
    factor2 = (x - x1)
    fq = factor1 / factor * fq1 + factor2 / factor * fq2
    return fq


def bilinear_interpolation(x, y, x1, x2, y1, y2, fq11, fq12, fq21, fq22):
    # fq11, fq12, fq21, fq22 are the valuese at the four points.
    factor = (x2 - x1) * (y2 - y1)
    factor11 = (x2 - x) * (y2 - y) / factor
    factor21 = (x - x1) * (y2 - y) / factor
    factor12 = (x2 - x) * (y - y1) / factor
    factor22 = (x - x1) * (y - y1) / factor
    fq = factor11 * fq11 + factor21 * fq21 + factor12 * fq12 + factor22 * fq22
    return fq


def scale_up(im_array, dst_shapes):
    dst_shapes = list(map(int, dst_shapes))
    src_height, src_width = im_array.shape[:2]
    dst_height, dst_width = dst_shapes[:2]
    src_x_mag = np.rint(np.linspace(0, dst_width - 1, src_width)).astype(int)
    src_y_mag = np.rint(np.linspace(0, dst_height - 1, src_height)).astype(int)
    src_x, src_y = np.meshgrid(np.arange(0, src_width),
                                            np.arange(0,  src_height))
    dst_x, dst_y = np.meshgrid(np.arange(0, dst_width),
                                            np.arange(0,  dst_height))
    src_x_m, src_y_m = np.meshgrid(src_x_mag, src_y_mag)
    im_array_new = np.zeros(dst_shapes)
    im_array_new[src_y_m, src_x_m] = im_array

    for i in range(len(src_x_mag)-1):
        if src_x_mag[i+1] - src_x_mag[i] > 1:
            for j in np.arange(src_x_mag[i]+1, src_x_mag[i+1]):
                im_array_new[:, j] = linear_interpolation(j, src_x_mag[i], src_x_mag[i + 1],
                                                          im_array_new[:, src_x_mag[i]],
                                                          im_array_new[:, src_x_mag[i+1]])

    for i in range(len(src_y_mag)-1):
        if src_y_mag[i+1] - src_y_mag[i] > 1:
            for j in np.arange(src_y_mag[i]+1, src_y_mag[i+1]):
                im_array_new[j, :] = linear_interpolation(j, src_y_mag[i], src_y_mag[i+1],
                                                          im_array_new[src_y_mag[i], :],
                                                          im_array_new[src_y_mag[i+1], :])

    dst_width_intp = np.setdiff1d(np.arange(0,  dst_width), src_x_mag)
    dst_height_intp = np.setdiff1d(np.arange(0,  dst_height), src_y_mag)

    for j in dst_width_intp:
        for i in dst_height_intp:
            im_array_new[i, j] = bilinear_interpolation(i, j, i-1, i+1, j-1, j+1,
                                                        im_array_new[i-1, j-1], im_array_new[i-1, j+1],
                                                        im_array_new[i+1, j-1], im_array_new[i+1, j+1])

    return im_array_new


def scale_down(im_array, dst_shapes):
    dst_shapes = list(map(int, dst_shapes))
    src_height, src_width = im_array.shape[:2]
    dst_height, dst_width = dst_shapes[:2]
    dst_x_mag = np.rint(np.linspace(0, src_width - 1, dst_width)).astype(int)
    dst_y_mag = np.rint(np.linspace(0, src_height - 1, dst_height)).astype(int)
    dst_x_m, dst_y_m = np.meshgrid(dst_x_mag, dst_y_mag)
    im_array_new = np.zeros(dst_shapes)
    im_array_new = im_array[dst_y_m, dst_x_m]
    return im_array_new


def main():
    parser = argparse.ArgumentParser(
        description="Use to resize an imgae")
    parser.add_argument("-i", "--inputfile", dest="inputfile",
                        required=True,
                        help="inputfile image file")
    parser.add_argument("-o", "--outfile", dest="outfile",
                        required=True,
                        help="output image file")
    parser.add_argument("-he", "--height", dest="height",
                        type=int,
                        help="Height")
    parser.add_argument("-w", "--width", dest="width",
                        type=int,
                        help="width ")
    parser.add_argument("-c", "--clobber",
                        action='store_true',
                        default=False,
                        help="Overwrite original file.( default:False)")
    args = parser.parse_args()

    img = Image.open(args.inputfile)
    im_array = np.array(img).astype(float)
    src_height = im_array.shape[0]
    src_width = im_array.shape[1]

    if args.width is not None and args.height is None:
        htimes = args.width / src_width
        wtimes = htimes
    elif args.width is None and args.height is not None:
        htimes = args.height / src_height
        wtimes = htimes
    elif args.width is not None and args.height is not None:
        htimes = args.height / src_height
        wtimes = args.width / src_width
    else:
        raise ValueError('Please input one parameter at least')

    if htimes >= 1 and wtimes >= 1:
        dst_shapes = [im_array.shape[0] * htimes, im_array.shape[1] * wtimes, 3]
        img_array_new = scale_up(im_array, dst_shapes)
    if htimes >= 1 and wtimes < 1:
        dst_shapes = [im_array.shape[0] * htimes, im_array.shape[1], 3]
        im_array = scale_up(im_array, dst_shapes)
        dst_shapes = [im_array.shape[0], im_array.shape[1] * wtimes, 3]
        img_array_new = scale_down(im_array, dst_shapes)
    if htimes < 1 and wtimes >= 1:
        dst_shapes = [im_array.shape[0] * htimes, im_array.shape[1], 3]
        im_array = scale_down(im_array, dst_shapes)
        dst_shapes = [im_array.shape[0], im_array.shape[1] * wtimes, 3]
        img_array_new = scale_up(im_array, dst_shapes)
    if htimes < 1 and wtimes < 1:
        dst_shapes = [im_array.shape[0] * htimes, im_array.shape[1] * wtimes, 3]
        img_array_new = scale_down(im_array, dst_shapes)

    img_new = Image.fromarray(np.uint8(img_array_new))
    if os.path.exists(args.outfile) and not args.clobber:
        raise FileExistsError(f'file already exists: {args.outfile}')
    else:
        img_new.save(args.outfile)


if __name__ == "__main__":
    main()
