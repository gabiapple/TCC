from perceptron import Perceptron

import sys


def main(args):
    p = Perceptron('/home/gabriela/projects/TCC/resultados/CurrentAlgorithms/Datasets/Sol#_Sol.csv', 2)
    p.get_nota([157, 209, 263, 313, 168, 210, 235, 259])
    p.get_nota([99,148,197,248,296,186,198,220])

if __name__ == "__main__":
    main(sys.argv)
