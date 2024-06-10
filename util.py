import numpy as np

def get_R(y,size,ROS):
    R = dict()

    if(ROS == 'NSHP'):
        NSHP_get_R(y,size,R)
    return R

def NSHP_get_R(y,size,R):
    for i in range(size+1):
        for j in range(size+1):
            if i == 0 and j == 0:
                R[(i,j)] = np.zeros((1,1))
            elif i == 0:
                R[(i,j)] = np.zeros((1,4*j))
            elif j == 0:
                R[(i,j)] = np.zeros((4*i,1))
            else:
                R[(i,j)] = np.zeros((4*i,4*j))
            
            for ii in range(R[(i,j)].shape[0]):
                for jj in range(R[(i,j)].shape[1]):


def NSHP_get_G_points(size):
    points = dict()
    points[0] = np.zeors((1,2))
    for i in range(1, size + 1):
        points[i] = NSHP_get_g_points(i)
    return points

def NSHP_get_g_points(size):
    points = np.empty((4 * size, 2))
    points[:size, 1] = size
    points[size : 3 * size, 0] = size
    points[3 * size : 4 * size, 1] = -size

    for j in range(size):
        points[j, 0] = j
        points[j + size, 1] = size - j
        points[j + 2 * size, 1] = -j
        points[j + 3 * size, 0] = size - j
    return points

# y 범위를 벗어나는 지점은 어떻게 할 것인가? white noise?
# rho(y-a , y-b)
def get_rho(y,a,b):
    
    while(True):
        

def rho_jkn(y,j,k,n,ROS):
    if ROS == 'NSHP': point = NSHP_get_g_points(n)

    g_size = point.shape[0]
    result = np.zeros((1,g_size))
    for i in range(g_size):
        result[i] = get_rho(y,i,j,k,ROS)