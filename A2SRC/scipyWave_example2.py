# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 15:09:04 2014

@author: Christian Muenker
"""

import numpy as np
from scipy.io import wavfile

rate, data = wavfile.read('D:/Daten/design/python/A2SRC/A2SRC/M1F1-float32WE-AFsp.wav') # returns np array with 16b integers
in_nbits = data.dtype
print(in_nbits)
print(np.max(data))
scale = 2**32 / 2**16
print(scale)

# Take the sine of each element in `data`.
scaleddata = data.astype(np.float32)/scale # returns float array

# Cast `scaled` to an array with a 16 bit signed integer data type.
newdata = scaleddata.astype(np.int16)

# Write the data to 'newname.wav'
wavfile.write('D:/Daten/design/python/A2SRC/A2SRC/newfile.wav', rate, newdata)

print(wavfile.KNOWN_WAVE_FORMATS)

