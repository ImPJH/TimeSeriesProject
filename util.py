import numpy as np

def get_R(y,size,ROS):
    R = dict()

    if(ROS == 'NSHP'):
        NSHP_get_R(y,size,R)
    return R

# TODO get R
def NSHP_get_R(y,size,R):
    points = NSHP_get_g_points(size)
    for i in range(size+1):
        for j in range(size+1):
            if i == 0 and j == 0:
                R[(i,j)] = np.ones((1,1))
            elif i == 0:
                R[(i,j)] = rho_jkn(y,0,0,j,'NSHP')
            elif j == 0:
                R[(i,j)] = np.zeros((4*i,1))
                for ii in range(4*i):
                    R[(i,j)][ii] = rho_jkn(y,points[ii,0],points[ii,1],0,'NSHP')
            else:
                R[(i,j)] = np.zeros((4*i,4*j))
                for ii in range(4*i):
                    R[(i,j)][ii] = rho_jkn(y,points[ii,0],points[ii,1],j,'NSHP')
                R[(i,j)] = np.array(R[(i,j)])
    print("get_R done!")



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
    return points.astype(int)

# y 범위를 벗어나는 지점은 어떻게 할 것인가? -> zero padding
# rho(y-a , y-b)
# sigma
def get_rho(y,j,k):
    sample1 = y
    sample2 = list()
    for row in range(y.shape[0]):
        tmp_row = list()
        for col in range(y.shape[1]):
            if row+j <y.shape[0] and col+k <y.shape[1]:
                tmp_row.append(y[row+j][col+k])
            else:
                tmp_row.append(0)
        sample2.append(tmp_row)
    
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    return np.mean((sample1-mean1)*(sample2-mean2))

def rho_jkn(y,j,k,n,ROS):
    if n == 0:
        result = np.array(get_rho(y,j,k))
    else:
        if ROS == 'NSHP':
            points = NSHP_get_g_points(n)

        g_size = points.shape[0]
        result = np.zeros((1,g_size))
        for i in range(g_size):
            result[0][i] = get_rho(y,j-points[i][0],k-points[i][1])
        result = np.array(result)
    return result

def get_sigma_00(y):
    return np.std(y)