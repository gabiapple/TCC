from numpy.fft import fft
from numpy.fft import fftfreq
from scipy.io import wavfile

import matplotlib.pyplot as plt
import sys

def plot(x, y):
    plt.plot(x, y)
    plt.grid()
    plt.show()

def get_fft(filepath):
    fs, data = wavfile.read(filepath)
    fourier  = fft(data)
    fa = round(fs/2)
    freq = fftfreq(fa, d=1/fa)

    fourier = abs(fourier)
    fourier = fourier/max(fourier)
    fourier = fourier[:fa]

    print(fourier)
    plot(freq, fourier)

    return fs, freq, fourier



def main(args):
    get_fft(args[1])

if __name__ == '__main__':
    main(sys.argv)
