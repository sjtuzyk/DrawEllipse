#!/usr/bin/env python3
#Author: Zhu Yong-kai (yongkai_zhu@hotmail.com)
import os
import argparse

import math
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


def draw_ellipse(shapes, center, a, b, phi):
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


def save_Fits(clobber,file_name, img):
    hdu = fits.PrimaryHDU(img)
    if os.path.exists(file_name):
        if clobber:
            os.remove(file_name)
            hdu.writeto(file_name)            
        else:
            print("Error: The file already exists.")
    else:
        hdu.writeto(file_name)


def main():
    parser = argparse.ArgumentParser(
        description="Ellipse rotation transformation")
    parser.add_argument("-o", "--outfile", dest="outfile",
                        required=True,
                        help="output Fits file")
    parser.add_argument("-or", "--orientation", dest="orientation",
                        type=float, 
                        required=True,
                        help="orientation")
    parser.add_argument("-a", "--axis1 ", dest="axis1",
                        type=float, 
                        required=True,
                        help="semi-major axes ")
    parser.add_argument("-b", "--axis2", dest="axis2",
                        type=float, 
                        required=True,
                        help="semi-minor axes")
    parser.add_argument("-cl", "--clobber", dest="clobber",
                        default=False,
                        type=bool,
                        help="Overwrite original file.( default:False)")
    parser.add_argument("-s", "--size ", dest="size",
                        type=int, 
                        required=True,
                        help="image size ")
    parser.add_argument("-l1", "--location1", dest="location1",
                        type=int,
                        required=True,
                        help="Location of center  on the X axis")
    parser.add_argument("-l2", "--location2", dest="location2",
                        type=int,
                        required=True,
                        help="Location of center  on the Y axis")
    args = parser.parse_args()

    a = args.axis1
    b = args.axis2
    phi = args.orientation
    size=args.size
    shapes = [size, size]
    center = [args.location1,args.location2]
    img = draw_ellipse(shapes, center, a, b, phi)
    save_Fits(args.clobber, args.outfile, img)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
