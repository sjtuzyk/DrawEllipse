#/usr/bin/env python3
#Author: Zhu Yong-kai (yongkai_zhu@hotmail.com)
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from astropy.io import fits
from matplotlib import pyplot as plt


class PSD:
    """
    Caculate the 2-dimension power spectral density.
    """
    def __init__(self, image, pixel=1.0, step=1):
        self.image = np.array(image, dtype = float)
        self.shape = self.image.shape
        self.pixel = pixel
        if self.shape[0] % 2 ==0:
            self.center = [self.shape[0] / 2, self.shape[1] / 2]
        else:
            self.center = [(self.shape[0] - 1) / 2, (self.shape[1] - 1) / 2]
        self.step = step

    def radii(self):
        shape = self.shape
        points1 = np.arange(0, shape[0])
        points2 = np.arange(0, shape[1])
        x, y = np.meshgrid(points1, points2)
        x_center, y_center = self.center
        distances = np.sqrt((x-x_center)**2+(y-y_center)**2)
        return distances

    def freq(self, r, d=1.0):
        n = self.shape[0]
        f = r / n / d
        return f

    def cal_psd2d(self):
        I = self.image
        Ifft2 = np.fft.fft2(I)
        Pxx_shift = np.fft.fftshift(Ifft2)
        Pxx = (abs(Pxx_shift))**2 / Pxx_shift.shape[0] / Pxx_shift.shape[1]\
            / self.pixel / self.pixel
        self.psd2d = Pxx
        return self.psd2d

    def cal_psd1d(self):
        OD = []
        nu = []
        rho = self.radii()
        Pxx = self.cal_psd2d()
        for r in np.arange(0, self.shape[0] / 2, self.step):
            r1 = r
            r2 = r + 1
            if r2 <= self.shape[0]:
                R = (rho >= r1).astype(int) * (rho < r2).astype(int)
                Pxx_R = R * Pxx
                nu.append(r)
                OD.append(np.median(Pxx_R[R.nonzero()[0],R.nonzero()[1]]))
        nu = np.array(nu)
        OD = np.array(OD)
        k = self.freq(nu)
        self.k = k
        self.psd1d = OD
        return self.k, self.psd1d

    def plot(self,outfile):
        """
        plot the 1-D radial power spectrum.
        """
        k = self.k
        OD = self.psd1d
        fig = plt.figure()
        plt.loglog(k, OD, '.-')
        plt.savefig(outfile)
        plt.show()

    def save(self, outfile):
        dataframe = pd.DataFrame({'k':self.k, 'K*K':self.psd1d})
        dataframe.to_csv(outfile, index=False, sep='\t')


def main():
    parser = argparse.ArgumentParser(
       description="Caculate the power spectrum distribution of\
       the input image")
    parser.add_argument("-i", "--inputfile", dest="inputfile",
                        required=True,
                        help="inputfile image file")
    parser.add_argument("-oc", "--outfile1", dest="outfile1",
                        required=True,
                        help="output CSV file")
    parser.add_argument("-oj", "--outfile2", dest="outfile2",
                        required=True,
                        help="output JPG file")
    parser.add_argument("-s", "--step", dest="step",
                        type=float,
                        help="sampling interval")
    parser.add_argument("-p", "--pixel", dest="pixel",
                        type=float,
                        help="pixel of the image")
    parser.add_argument("-c", "--clobber",
                        action='store_true',
                        default=False,
                        help="Overwrite original file.( default:False)")
    args = parser.parse_args()
    for i in [args.outfile1, args.outfile2]:
        if os.path.exists(i) and not args.clobber:
            raise FileExistsError(f'file already exists:{i}')

    img = fits.open(args.inputfile)
    inputfile='cluster.fits'
    img = fits.open(inputfile)
    img = img[0].data
    psd = PSD(image=img, pixel=args.pixel, step=args.step)
    psd.cal_psd1d()
    psd.plot(args.outfile2)
    psd.save(args.outfile1)



if __name__=="__main__":
    main()

