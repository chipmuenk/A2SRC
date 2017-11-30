# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 09:14:47 2015

@author: Christian Muenker
"""

import numpy as np

# You can use either matplotlib or vispy to render this example:
#import matplotlib.pyplot as plt
import vispy.mpl_plot as plt

from vispy.io import read_png, load_data_file

n = 200
freq = 10
fs = 100.
t = np.arange(n) / fs
tone = np.sin(2*np.pi*freq*t)
noise = np.random.RandomState(0).randn(n)
signal = tone + noise
magnitude = np.abs(np.fft.fft(signal))
freqs = np.fft.fftfreq(n, 1. / fs)
flim = n // 2

# Signal
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.imshow(read_png(load_data_file('pyplot/logo.png')))
#
ax2 = fig.add_subplot(312)
ax2.plot(t, signal, 'k-')

# Frequency content
ax3 = fig.add_subplot(313)
idx = np.argmax(magnitude[:flim])
ax3.text(freqs[idx], magnitude[idx], 'Max: %s Hz' % freqs[idx],
        verticalalignment='top')
ax3.plot(freqs[:flim], magnitude[:flim], 'k-o')

plt.draw()

# NOTE: show() has currently been overwritten to convert to vispy format, so:
# 1. It must be called to show the results, and
# 2. Any plotting commands executed after this will not take effect.
# We are working to remove this limitation.

if __name__ == '__main__':
    fig.show(True)