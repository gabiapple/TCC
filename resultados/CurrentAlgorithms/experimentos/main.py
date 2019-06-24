from fft import get_fft
from ga import run_ga

import sys


def main(args):
    fs, freq, fourier = get_fft(args[1])
    print(run_ga(fourier))

if __name__ == "__main__":
    main(sys.argv)
