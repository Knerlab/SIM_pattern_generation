# -*- coding: utf-8 -*-
__author__ = "Yang Liu, Ming Song and Peter A. Kner"
__copyright__ = "Copyright 2023, SISO-SPIM Project"
__credits__ = ["Ming Song", "Yang Liu", "Peter A. Kner"]
__license__ = "GPL"
__version__ = "1.1.0"
__maintainer__ = "Ming Song"
__email__ = "ming.song@uga.edu"
__status__ = "Production"



import numpy as np
from numpy import matlib as mb
fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftn = np.fft.fftn
ifftn = np.fft.ifftn
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift
import tifffile as tf
from scipy.ndimage import gaussian_filter
from PIL import Image
from scipy.special import jn
from pdb import set_trace as st
import matplotlib.pyplot as plt
import os



class generate(object):
    
    def __init__(self):
        self.na1 = 0.425
        self.na2 = 0.457
        self.na_p = (self.na1 + self.na2) / 2
        # self.na_p = 0.28
        self.wl = 0.488
        self.nx = 512
        self.dx = self.wl/self.na2/2/8
        self.dp = 1/(self.nx*self.dx)
        self.n2 = 1.333
        self.p = self.dp*np.arange((-self.nx/2),(self.nx/2))
        p = self.p
        self.kx,self.ky = np.meshgrid(p,p,sparse=True)
        self.rho = np.sqrt(self.kx**2+self.ky**2)
        self.msk = self.rho<=self.na2/self.wl
        self.bpp2 = np.array(self.msk).astype('int64')
        self.bpp1 = np.array(self.rho<=self.na1/self.wl).astype('int64')
        self.move = int(self.na_p/self.wl/self.dp)
        self.move1 = int(self.move*np.cos(np.pi/3))+1
        self.move2 = int(self.move*np.sin(np.pi/3))+1
        self.annulus = self.bpp2-self.bpp1
        self.dz = .100
        self.zr = 25.5 # in microns
        self.Np = 50 # number of particles
        self.imgf = np.zeros((self.nx,self.nx), dtype=np.float32)
        lim1 = -self.zr/2
        lim2 = self.zr/2
        self.nz = int((lim2-lim1)/self.dz+1)
        # self.Zreleigh = 
        # np.pad use wrap to expand the image
   
    def getSLMimg(self,img,alpha,num_phase=5,pattern_px=20):



        save_path = f"{pattern_px}pixel_{num_phase}phase_lattice_squre"
        # Check whether the specified path exists or not
        isExist = os.path.exists(save_path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(save_path)
            print(f"{save_path} directory is created!")

        


        nx = 4096
        ny = 4096
        slmwidth = 2048
        slmheight = 1536
        pdx = int((nx-self.nx)*.5)
        pdy = int((ny-self.nx)*.5)
        I = np.pad(img,((pdy,pdy),(pdx,pdx)),'wrap')
        I = I/np.max(I)*255
 
        slm_h_start = int((nx-slmwidth)*.5)
        slm_v_start = int((nx-slmheight)*.5)
        move = int(pattern_px/num_phase)
        # st()
        for i in range(num_phase):
            slm_h=slm_h_start+i*move
            slm_v=slm_v_start
  
            temp = I[slm_v:slm_v+slmheight,slm_h:slm_h+slmwidth]
            temp[temp<=np.max(I)*alpha]=0
            temp[temp!=0]=255
            temp = temp.astype(np.uint8)
            final = Image.fromarray(temp)
            final.save('converted_pattern.bmp')
            im = Image.open('converted_pattern.bmp')
            im=im.convert('1')
            im.save(os.path.join(save_path,f'phase{i}_pattern_px{pattern_px}_1bit_squre.bmp'))
            im.close()

        tf.imshow(temp)
        #binary
        return temp
    # def getfivephase(self,I):
        
        
    def convertimg(self,filepath):
        im = Image.open(filepath)
        im=im.convert('1')
        im.save('1bitimg_converted.bmp')
        im.close()



    def getSquareLatticePixel(self, Npix = 10, phase = 0):
        ''' repeats pattern every Npix
            phase is in pixels '''
        k = 2*np.pi/self.wl
        x = np.linspace(-self.nx*self.dx*.5,self.nx*self.dx*.5,self.nx) #FOV_x in micron
        y = np.linspace(-self.nx*self.dx*.5,self.nx*self.dx*.5,self.nx) #FOV_y
        [xx,yy] = np.meshgrid(x,y)
        xxc = phase*self.dx
        rho = np.sqrt((xx-xxc)**2+yy**2)
        ## light sheet thickness
        fwhm = 5.0
        ls_sigma = fwhm/(2*np.log(2))
        ## bessel function maxima 1st: 7.0156, 2nd: 13.3237
        xmax = 7.0156
        na_p = xmax/(k*Npix*self.dx)
        bconf = self.dx*xmax# bessel confinement factor
        print(na_p)
        e1 = jn(0,k*rho*na_p)*np.exp(-0.5*(rho/bconf)**2)
        # st()
        tf.imshow(e1)
        spacing = Npix*self.dx
        ## create lattice
        st_x= int((self.nx)*.5)
        a1 = np.array([0, Npix])
        a2 = np.array([Npix, 0])
        elatt = np.zeros([self.nx,self.nx])
        for m in np.arange(-8,8):
            for n in np.arange(-8,8):
                b = m*a1 + n*a2
                C = np.roll(e1,b[0],axis=0)
                C = np.roll(C,b[1],axis=1)
                elatt = elatt + C
        # pattern
        I = np.abs(elatt)**2
        I = I/np.max(I)
        tf.imshow(I)
        # back pupil
        ls = np.exp(-0.5*(yy/ls_sigma)**2)
        # tf.imshow((np.abs(ls*elatt))**2)
        bpp = np.abs(fftshift(fft2(ls*elatt)))**2
        tf.imshow(bpp)
        elatt = elatt[st_x-Npix:st_x+Npix,st_x-Npix:st_x+Npix]
        I = np.abs(elatt)**2
        I = I/np.max(I)

        
        return I






def main():
    gen = generate()
    I=gen.getSquareLatticePixel(Npix=9)
    final = gen.getSLMimg(I,0.35,num_phase=3,pattern_px=9)
    plt.show()

    
    


if __name__ == '__main__':
    main()



     
        
        
        
        
        
        
        
        