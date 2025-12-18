from abc import ABC, abstractmethod
from functools import partial

import jax; jax.config.update('jax_enable_x64',True)
import jax.numpy as jp
import jax.scipy

import numpyro.distributions

import numpy as np

import inspect
import warnings

import time
import glob
import dill
import os

from astropy.io import fits

# Load image
# --------------------------------------------------------
def _img_loader(img,idx=0):
    if   isinstance(img,(fits.ImageHDU,fits.PrimaryHDU)):
        return img
    elif isinstance(img,fits.hdu.hdulist.HDUList):
        return img[idx]
    elif isinstance(img,str):
        img = fits.open(img)
        return img[idx]
    else:
        raise ValueError('img must be an ImageHDU or a string')
