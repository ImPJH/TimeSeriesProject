import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CausalAr import AR_2D
from util import CalBeta
from Findb import Findb

def test(y):
    # Read residual_2.csv
    df2 = pd.read_csv('source/residual_2.csv', header=None)
    sample = df2.values
    print('residual shape', sample.shape)

    size = 16
    # sample = np.random.rand(size*2,size*2) * 255

    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # ax.scatter(np.arange(128),np.arange(128),sample)
    # plt.savefig('random_sample.jpg')
    # plt.close()
    # print(sample)
    BetaCalculator = CalBeta(sample,size,ROS='NSHP')
    sigma_00 = BetaCalculator.get_sigma_00()
    R = BetaCalculator.get_R()   
    ar_2d_model = AR_2D(R,size,sigma_00)
    (beta, Ssigma) = ar_2d_model.run()
    print('BETA \n', beta)

    # plot AIC, BIC
    ar_2d_model.plot_AIC_BIC(sample)
    plt.show()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--algorithm", default="PPO", type=str)
    # parser.add_argument("--load", default=0, type=int)
    # parser.add_argument("--save", default=False, type=bool)
    y=None
    test(y)