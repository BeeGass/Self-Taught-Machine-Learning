import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math
import sys


def NormalDistOne():
    mu = 3
    sigma = math.sqrt(1)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 10)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()

def NormalDistTwo():
    mu = 3
    sigma = math.sqrt(10)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()

def NormalDistThree():
    mu = 3
    sigma = math.sqrt(10)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 1000)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))
    plt.show()

def main(argv):
    print(argv)
    if int(argv) == 0:
        print("running DistOne")
        NormalDistOne()
    elif int(argv) == 1:
        print("running DistTwo")
        NormalDistTwo()
    elif int(argv) == 2:
        print("running DistThree")
        NormalDistThree()


if __name__ == "__main__":
    main(sys.argv[1])
