import numpy as np

class CalBeta():
    def __init__(self,y, size, ROS = 'NSHP'):
        self.R = dict()
        self.y = y
        self.size = size
        self.ROS = ROS
        self.sigma_00 = self.get_sigma_00()
        self.g_points_dict = dict()
        self.rho_dict = dict()

    def get_R(self):
        if(self.ROS == 'NSHP'):
            self.NSHP_get_R()
        print("Get R done!")
        return self.R

    # TODO get R
    def NSHP_get_R(self):
        for i in range(self.size+1):
            points = self.NSHP_get_g_points(i)
            for j in range(self.size+1):
                if i == 0 and j == 0:
                    self.R[(i,j)] = np.ones((1,1))
                elif i == 0:
                    self.R[(i,j)] = self.rho_jkn(0,0,j)
                elif j == 0:
                    self.R[(i,j)] = np.zeros((4*i,1))
                    for ii in range(4*i):
                        self.R[(i,j)][ii] = self.rho_jkn(points[ii,0],points[ii,1],0)
                else:
                    self.R[(i,j)] = np.zeros((4*i,4*j))
                    for ii in range(4*i):
                        self.R[(i,j)][ii] = self.rho_jkn(points[ii,0],points[ii,1],j)
                    self.R[(i,j)] = np.array(self.R[(i,j)])

    def NSHP_get_G_points(self):
        points = dict()
        points[0] = np.zeors((1,2))
        for i in range(1, self.size + 1):
            points[i] = self.NSHP_get_g_points(i)
        return points

    def NSHP_get_g_points(self,s):
        if s in self.g_points_dict.keys():
            return self.g_points_dict[s]
        else:
            points = np.empty((4 * s, 2))
            points[:s, 1] = s
            points[s : 3 * s, 0] = s
            points[3 * s : 4 * s, 1] = -s

            for j in range(s):
                points[j, 0] = j
                points[j + s, 1] = s - j
                points[j + 2 * s, 1] = -j
                points[j + 3 * s, 0] = s - j
            self.g_points_dict[s] = points.astype(int)
            return self.g_points_dict[s]

    # y 범위를 벗어나는 지점은 어떻게 할 것인가? -> zero padding
    # rho(y-a , y-b)
    # sigma
    def get_rho(self,j,k):
        j = abs(j)
        k = abs(k)
        if (j,k) in self.rho_dict.keys():
            return self.rho_dict[(j,k)]
        else:
            sample1 = self.y
            sample2 = list()
            for row in range(self.y.shape[0]):
                tmp_row = list()
                for col in range(self.y.shape[1]):
                    if row+j <self.y.shape[0] and col+k <self.y.shape[1]:
                        tmp_row.append(self.y[row+j][col+k])
                    else:
                        tmp_row.append(0)
                sample2.append(tmp_row)
            
            mean1 = np.mean(sample1)
            mean2 = np.mean(sample2)
            self.rho_dict[(j,k)] = np.mean((sample1-mean1)*(sample2-mean2)) / self.sigma_00
            return np.mean((sample1-mean1)*(sample2-mean2)) / self.sigma_00

    def rho_jkn(self,j,k,n):
        if n == 0:
            result = np.array(self.get_rho(j,k))
        else:
            if self.ROS == 'NSHP':
                points = self.NSHP_get_g_points(n)

            g_size = points.shape[0]
            result = np.zeros((1,g_size))
            for i in range(g_size):
                result[0][i] = self.get_rho(j-points[i][0],k-points[i][1])
            result = np.array(result)
        return result

    def get_sigma_00(self):
        return np.std(self.y)
    
    def reset(self):
        self.rho_jkn_dict = dict()
        self.rho_dict = dict()
        self.g_points_dict = dict()