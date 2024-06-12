import argparse
import numpy as np
import matplotlib.pyplot as plt
from CausalAr import AR_2D
from util import CalBeta
from Findb import Findb

def test(y):
    size = 16
    sample = np.random.rand(size*2,size*2)
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
    print(beta)
    print(Ssigma)


    
    


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--algorithm", default="PPO", type=str)
    # parser.add_argument("--load", default=0, type=int)
    # parser.add_argument("--save", default=False, type=bool)
    y=None
    test(y)