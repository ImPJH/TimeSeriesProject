import argparse
import util
from CausalAr import AR_2D

def test(y):
    size = 64
    sigma_00 = util.get_sigma_00(y)
    R = util.get_R(y,size,'NSHP')
    AR_2D(R,size,sigma_00)
    (beta, Ssigma) = AR_2D.run()

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--algorithm", default="PPO", type=str)
    # parser.add_argument("--load", default=0, type=int)
    # parser.add_argument("--save", default=False, type=bool)
    test(y)