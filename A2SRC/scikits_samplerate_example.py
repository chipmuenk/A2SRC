# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 13:29:51 2014

Demonstration for the samplerate module that is wrapper around the 
Secret Rabbit Code from Erik de Castro Lopo (http://www.mega-nerd.com/SRC/)
offering high quality real-time resampling.
"""

import numpy as np
import matplotlib.pyplot as plt
import scikits.samplerate as SRC
#from scikits.samplerate import resample

fs = 44100.
fr = 48000.
# Signal to resample
sins = np.sin(2*np.pi*1000/fs*np.arange(0, fs*2))
# Ideal resampled signal
idsin = np.sin(2*np.pi*1000/fr*np.arange(0, fr*2))
#'zero_order_hold', 'linear', 'sinc_fastest', _medium and_best
conv1 = SRC.resample(sins, fr/fs, 'linear') 
conv3 = SRC.resample(sins, fr/fs, 'sinc_best')

err1 = conv1[fr:fr+2000] - idsin[fr:fr+2000]
err3 = conv3[fr:fr+2000] - idsin[fr:fr+2000]

plt.subplot(3, 1, 1)
plt.plot(idsin[fs:fs+2000])
plt.title('Resampler residual quality comparison')

plt.subplot(3, 1, 2)
plt.plot(err1)
plt.ylabel('Err. Linear')

plt.subplot(3, 1, 3)
plt.plot(err3)
plt.ylabel('Err. Sinc')
#plt.savefig('example1.png', dpi = 100)
plt.show()