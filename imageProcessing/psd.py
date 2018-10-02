#/usr/bin/env python3
#Author: Zhu Yong-kai (yongkai_zhu@hotmail.com)
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from astropy.io import fits

import matplotlib
import matplotlib.style
from  matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

#matplotlib settings
matplotlib.style.use("ggplot")
Params = {"font.family" : "serif",
          "xtick.major.size" : 7.0,
          "xtick.major.width" : 2.0,
          "xtick.minor.size" : 4.0,
          "xtick.minor.width" : 1.5,
          "ytick.major.size" : 7.0,
          "ytick.major.width" : 2.0,
          "ytick.major.size" : 4.0,
          "ytick.minor.width" : 1.5,
         }
for keys in Params.keys():
    matplotlib.rcParams[keys] = Params[keys]


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

    def plot(self, ax):
        """
        plot the 1-D radial power spectrum.
        """
        k = self.k
        psd1d = self.psd1d
        ax.plot(k, psd1d, marker="o", color="blue", label="median")
        xmin, xmax = k[1] / 1.2, k[-1] * 1.1
        ymin, ymax = psd1d.min()/10, psd1d.max()*1.5
        ax.set(xscale="log", yscale="log",
               ylim= (ymin, ymax), xlim=(xmin,xmax))
        ax.set_xlabel(r"$k$ [%s$^{-1}$]" % "pixel",fontsize=16)
        ax.set_ylabel('power',fontsize=16)
        ax.set_title("Radial Power Spectral Density", size=16, weight="bold")
        ax.legend()
        return ax


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
    parser.add_argument("-o", "--outfile", dest="outfile",
                        required=True,
                        help="output CSV file")
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
    if os.path.exists(args.outfile) and not args.clobber:
        raise FileExistsError(f'file already exists:{args.outfile}')

    img = fits.open(args.inputfile)
    img = img[0].data
    psd = PSD(image=img, pixel=args.pixel, step=args.step)
    psd.cal_psd1d()
    fig = Figure(figsize=(8,8), dpi=150)
    FigureCanvas(fig)
    ax = fig.add_subplot(1,1,1)
    psd.plot(ax = ax)
    plotfile = os.path.splitext(args.outfile)[0] + ".png"
    fig.savefig(plotfile)
    psd.save(args.outfile)



if __name__=="__main__":
    main()

