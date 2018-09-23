#!/usr/bin/env python3
#Author: Zhu Yong-kai (yongkai_zhu@hotmail.com)
import os
import argparse

import math
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits



def is_exists(file_name):
    n=[0]
    if os.path.exists(file_name):
        print("Given file path is exist. ")
       # override_rename = raw_input("Override file or Rename it. (r/o):")
        override_rename = 'r'
        if override_rename == 'r':
            file_name_new=os.path.splitext(file_name)[0]+str(n[0])+'.fits'
            while os.path.exists(file_name_new):
                n[0]+=1
                file_name_new=os.path.splitext(file_name)[0]+str(n[0])+'.fits'
        else:
            os.remove(file_name)
    else:
        file_name_new = file_name
    return file_name_new
    

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


def save_Fits(file_name, img):
    file_name=is_exists(file_name)
    hdu = fits.PrimaryHDU(img)
    hdu.writeto(str(file_name))


def main():
    parser = argparse.ArgumentParser(
        description="Ellipse rotation transformation")
    parser.add_argument("-o", "--outfile", dest="outfile",
                        default="ERT.fits",
                        help="output Ellipse Fits file " +
                        "(default: ERT.fits)")
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
    args = parser.parse_args()

    a = args.axis1
    b = args.axis2
    phi = args.orientation
    shapes = [2.5*a, 2.5*a]
    center = [2.5*a/2, 2.5*a/2]
    img = draw_ellipse(shapes, center, a, b, phi)
    save_Fits(args.outfile, img)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    main()
