from numpy.fft import fft
from numpy.fft import fftfreq
from scipy.io import wavfile

from run import write_csv


import matplotlib.pyplot as plt
import numpy as np
import sys

def plot(x, y):
    plt.plot(x, y)
    plt.grid()
    plt.show()

def fft_slot(freq):
    pass


def get_fft(filepath):
    fs, som = wavfile.read(filepath)
    length_som = len(som)
    half_lenght_som = round(length_som/2)
    fourier  = abs(fft(som))
    fourier = fourier/max(fourier)
    fourier = fourier[:half_lenght_som]

    freq = fftfreq(len(som), d=1/fs)
    freq = freq[:half_lenght_som]

    j = 0
    soma = 0
    respfreq = []
    for i in range(len(freq) - 1):
        if(round(freq[i]) == round(freq[i+1])):
           soma = fourier[i+1] + soma
           j = j + 1
        else:
           respfreq.append(soma/(j+1))
           j = 0
           soma = fourier[i+1]

    respfreq_size = int(fs/4)
    return fs, freq, fourier, respfreq[:respfreq_size]

def get_max_values_range(data, step, fstart, N):
    start_f=125
    end_f = 250
    maximus = []
    maximus2 = []
    maximus2val = []
    for i in range(fstart,N,step):
        f = np.where(data == np.amax(data[i:i+step-1]))[0][0]
        if not f:
           continue
        if f in maximus:
           continue
        newf = f

        # Convert values to a frequency range
        while newf < start_f:
           newf = newf*2
        while newf > end_f:
           newf = newf/2

        newf = int(newf)
        f = int(f)

        # Skip repeated or similar frequencies
        frequency_exist = False
        for k in range(4):
            if (newf + k) in maximus2 or (newf - k) in maximus2:
                frequency_exist = True
                break
        if frequency_exist:
            continue

        maximus2.append(newf)
        maximus2val.append(data[f])
        maximus.append(f)
    maximus.sort()
    maximus2.sort()
    maximus2val.sort()
    return maximus, maximus2, maximus2val

def main(args):
    acordes = [
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Do#_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Re#_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Mi_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Fa#_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Sol#_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/La#_18.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_1.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_2.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_3.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_4.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_5.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_6.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_7.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_8.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_9.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_10.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_11.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_12.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_13.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_14.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_15.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_16.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_17.wav',
        '/home/aluno/TCC/resultados/GuitarChords/Variações/Si_18.wav'
    ]
    all_frequencies = []
    max_atributes = 5
    variation_per_chord = 9
    i = 0
    j = 0
    while i  < len(acordes):
        chord = j + 1
        k = i + 1
        chord_frequencies = []
        print(chord, (k + variation_per_chord - 1))
        while(i < (k + variation_per_chord - 1)):
            print(i, acordes[i])
            fs, freq, fourier, respfreq = get_fft(acordes[i])
            _, frequencies, _ = get_max_values_range(respfreq, 100, 65, 1000)
            frequencies = frequencies[:max_atributes]
            frequencies.extend([chord])
            chord_frequencies.append(frequencies)
            i = i + 1
        j = j + 1
        all_frequencies.extend(chord_frequencies)
    dataset = [["Freq1", "Freq2", "Freq3", "Freq4", "Freq5", "Chord"]]
    dataset.extend(all_frequencies)
    write_csv('NewDatasets/acordes_maiores_216.csv', dataset)

if __name__ == '__main__':
    main(sys.argv)
