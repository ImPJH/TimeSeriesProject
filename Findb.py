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
        self.BetaCalculator = CalBeta(y, self.l, ROS = 'NSHP')
        self.construct_b_mat()
    
    def y_shift(self,j,k):
        result = np.zeros_like(self.y)
        (row_size, col_size) = self.y.shape
        for row in range(row_size):
            target_row = row - k
            for col in range(col_size):
                target_col = col - j
                if(target_col >= 0 and target_row >= 0 and target_col < self.y.shape[1] and target_row < self.y.shape[0]):
                    result[row][col] = self.y[target_row][target_col]
        return result
    
    def construct_b_mat(self):
        if self.ROS == 'NSHP':
            b = np.zeros((2*self.l+1,self.l))
            points = self.BetaCalculator.NSHP_get_G_points()
            b[self.l][0] = self.beta[(self.l,0)][0][0]
            for i in range(1,len(points)):
                for p in range(4*i):
                    ss = points[i][p,0]
                    tt = points[i][p,1]
                    b[self.l-tt][ss] = self.beta[(self.l,i)][p][0]
        return b

    def run(self):
        init_b = self.construct_b_mat()
        _lambda = 0.2
        b = cp.Variable(init_b.shape)
        b.value = init_b
        sum_obj = 0
        obj = 0
        points = self.BetaCalculator.NSHP_get_G_points()
        sum_obj = b[self.l][0]*self.y
        for i in range(1,len(points)):
            for p in range(4*i):
                ss = points[i][p,0]
                tt = points[i][p,1]
                sum_obj += b[self.l-tt][ss]*self.y_shift(ss,tt)

        obj += cp.sum(cp.power(cp.multiply(cp.power(self.y-sum_obj,2),cp.inv_pos(self.y)),2))
        obj += _lambda*cp.sum(cp.abs(b))

        obj = cp.Minimize(obj)

        print("problem setting")
        print("before_solving\n",b.value)
        prob = cp.Problem(obj)
        result = prob.solve()

        print("after_solving\n",b.value)


