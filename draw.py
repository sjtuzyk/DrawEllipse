#!/usr/bin/env python3
#Author: Zhu Yong-kai (yongkai_zhu@hotmail.com)

import matplotlib.pyplot as plt
import numpy as np
import math
from astropy.io import fits
import argparse


def EllipseShape(shapes, center, a, b, phi):
    phi = math.radians(phi)
    points1 = np.arange(0, shapes[0])
    points2 = np.arange(0, shapes[1])
    x, y = np.meshgrid(points1, points2)
    x_center, y_center = center
    x_rotation = (x-x_center)*np.cos(phi)+(y-y_center)*np.sin(phi)
    y_rotation = (x-x_center)*np.sin(phi)-(y-y_center)*np.cos(phi)
    distances = ((x_rotation)**2)/(a**2)+((y_rotation)**2)/(b**2)
    img = (distances <= 1).astype(int)
    return img


def Img2Fits(filename, img):
    hdu = fits.PrimaryHDU(img)
    hdu.writeto(str(filename))


def main():
    parser = argparse.ArgumentParser(
        description="Ellipse rotation transformation")
    parser.add_argument("-o", "--outfile", dest="outfile",
                        default="ERT.fits",
                        help="output Ellipse Fits file " +
                        "(default: ERT.fits)")
    parser.add_argument("-or", "--orientation", dest="orientation",
                        type=float, default=60,
                        help="orientation")
    parser.add_argument("-a", "--axis1 ", dest="axis1",
                        type=float, default=60,
                        help="long semi-axis ")
    parser.add_argument("-b", "--axis2", dest="axis2",
                        type=float, default=40,
                        help="short semi-axis")

    args = parser.parse_args()

    a = args.axis1
    b = args.axis2
    phi = args.orientation
    shapes = [150, 150]
    center = [60, 60]
    a = 60
    b = 40
    img = EllipseShape(shapes, center, a, b, phi)
    Img2Fits(args.outfile, img)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
