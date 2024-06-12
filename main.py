import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from CausalAr import AR_2D
from util import CalBeta
from Findb import Findb

def test(y):
    size = 10
    # sample = np.random.rand(64,64)

    # y = sample

    # fig = plt.figure()
    # ax = fig.add_subplot(111,projection='3d')
    # ax.scatter(np.arange(128),np.arange(128),sample)
    # plt.savefig('random_sample.jpg')
    # plt.close()
    # print(sample)

    # normalize
    # min_y = min(y)
    # max_y = max(y)
    # y = (y-min_y)/(max_y-min_y)
    BetaCalculator = CalBeta(y,size,ROS='NSHP')
    sigma_00 = BetaCalculator.get_sigma_00()
    R = BetaCalculator.get_R()   
    ar_2d_model = AR_2D(R,size,sigma_00)
    (beta, Ssigma) = ar_2d_model.run()
    print('BETA \n', beta)

    # # plot AIC, BIC
    ar_2d_model.plot_AIC_BIC(y)
    # plt.show()

    noncausal_solver = Findb(y,size,beta,ROS='NSHP')
    noncausal_solver.run()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--algorithm", default="PPO", type=str)
    # parser.add_argument("--load", default=0, type=int)
    # parser.add_argument("--save", default=False, type=bool)
    # Read residual_2.csv
    df2 = pd.read_csv('source/residual_128.csv', header=None)
    y = df2.values
    print('residual shape', y.shape)
    test(y)