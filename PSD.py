#/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from astropy.io import fits
from matplotlib import pyplot as plt

def annulus(shapes, center, r1, r2):
    points1 = np.arange(0, shapes[0])
    points2 = np.arange(0, shapes[1])
    x, y = np.meshgrid(points1, points2)
    x_center, y_center = center
    distances = np.sqrt((x-x_center)**2+(y-y_center)**2)
    ann = (distances >= r1).astype(int) * (distances < r2).astype(int)
    return ann


def main():
    parser = argparse.ArgumentParser(
        description="Caculate the power spectrum distribution of\
        the input imgae")
    parser.add_argument("-i", "--inputfile", dest="inputfile",
                        required=True,
                        help="inputfile image file")
    parser.add_argument("-oc", "--outfile1", dest="outfile1",
                        required=True,
                        help="output CSV file")
    parser.add_argument("-oj", "--outfile2", dest="outfile2",
                        required=True,
                        help="output JPG file")
    parser.add_argument("-c", "--clobber",
                        action='store_true',
                        default=False,
                        help="Overwrite original file.( default:False)")
    args = parser.parse_args()


    hdu1 = fits.open(args.inputfile)
    I = hdu1[0].data
    Ifft2 = np.fft.fft2(I)
    Pxx_shift = np.fft.fftshift(Ifft2)
    Pxx = (abs(Pxx_shift))**2 / Pxx_shift.shape[0] / Pxx_shift.shape[1]
    Pfreq = np.fft.fftfreq(Pxx.shape[0])
    OD = []
    nu = []
    shapes = Pxx.shape
    center = [shapes[0] / 2, shapes[1] / 2]
    r0 = 1
    delta_r = 1
    for r in np.arange(r0, min(shapes) / 2, delta_r):
        r1 = r - delta_r
        r2 = r
        R = annulus(shapes, center, r1, r2)
        Pxx_R = R * Pxx
        nu.append(r)
        OD.append(np.median(Pxx_R[R.nonzero()[0],R.nonzero()[1]]))

    k = np.array(nu) / shapes[0]
    dataframe = pd.DataFrame({'k':k, 'Pk':OD})
    dataframe.to_csv(args.outfile1, index=False, sep='\t')
    plt.loglog(k, OD, '.')
    plt.savefig(args.outfile2)
    plt.show()


if __name__=="__main__":
    main()

