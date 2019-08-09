from numpy.fft import fft
from numpy.fft import fftfreq
from scipy.io import wavfile


import matplotlib.pyplot as plt
import numpy as np
import sys

def plot(x, y):
    plt.plot(x, y)
    plt.grid()
    plt.show()

def get_fft(filepath):
    fs, data = wavfile.read(filepath)
    fourier  = fft(data)
    fa = round(fs*2)
    freq = abs(fftfreq(fa, d=1/fa))

    fourier = abs(fourier)
    fourier = fourier/max(fourier)
    fourier = fourier[:fa]

    #plot(freq, fourier)
#    plot(freq[0:1000], fourier[0:1000])

    return fs, freq, fourier

def get_max_values_range(data, step, fstart, N):
    start_f=125
    end_f = 250
    maximus = []
    maximus2 = []
    maximus2val = []
    print(list(range(fstart,N,step)))
    for i in range(fstart,N,step):
        f = np.where(data == np.amax(data[i:i+step-1]))[0][0]
        newf = f
        while newf < start_f:
            cof = round(start_f/f)
            if cof % 2 == 1:
                cof = cof + 1
            newf = int(newf*cof)
        while newf > end_f:
            cof = round(newf/end_f)
            if cof % 2 == 1:
                cof = cof + 1
            newf = int(newf/cof)

        print("F: {} newf: {}".format(f, newf))
        maximus2.append(newf)
        maximus2val.append(data[newf])
        maximus.append(f)

    return maximus, maximus2, maximus2val

def main(args):
    fs, freq, fourier = get_fft(args[1])
    _, frequencies, _ = get_max_values_range(fourier, 100, 65, 1000)
    print(frequencies)

if __name__ == '__main__':
    main(sys.argv)
