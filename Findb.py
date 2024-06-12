import numpy as np
import cvxpy as cp
import math
from util import CalBeta

class Findb():
    def __init__(self,y,m,beta,ROS='NSHP'):
        self.y = y
        self.beta = beta
        self.l = math.floor((m+1)/2)
        self.ROS = ROS
        self.BetaCalculator = CalBeta(y, m, ROS = 'NSHP')
        self.construct_b_mat()
    
    def y_shift(self,j,k):
        result = np.zeros_like(self.y)
        (row_size, col_size) = self.y.shape
        for row in range(row_size):
            target_row = row - j
            for col in range(col_size):
                target_col = col - k
                if(target_col >= 0 and target_row >= 0):
                    result[target_row][target_col] = self.y[row][col]
        return result
    
    def construct_b_mat(self):
        if self.ROS == 'NSHP':
            self.b = np.zeros((2*self.m+1,self.m))
            points = self.BetaCalculator.NSHP_get_G_points()
            self.b[0][0] = self.beta[(self.m,0)][0][0]
            for i in range(1,len(points)):
                for p in range(4*i):
                    row = points[i][p,0]
                    col = points[i][p,1]
                    self.b[row][col] = self.beta[(self.m,i)][p][0]

    def run(self):
        self.y = cp.Variable()


