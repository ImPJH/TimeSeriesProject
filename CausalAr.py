import numpy as np

# TODO: R을 구하는 코드가 필요
# Todo2: debugging이 필요
# Todo3: optimize가 필요

class AR_2D:
    def __init__(self, R, size, sigma_00):
        self.R = R
        self.size = size
        self.sigma_00 = sigma_00

        self.Theta = dict()
        self.Theta_inv = dict()
        self.h = dict()
        self.beta = dict()
        self.Phi = dict()
        self._lambda = np.zeros(self.size)
        self.Ssigma = np.zeros(self.size)

        self.R11_inv = np.linalg.inv(self.R[(1, 1)])

        # beta_m(0)는 항상 -1
        self.beta[1] = np.array([-1, self.R11_inv @ self.R[(1, 0)]])

        self.Phi[(2, 1)] = np.array([self.R11_inv @ self.R[(1, 2)]])

        # lambda는 pdf에 나온 index에서 2를 뺀 index로 사용
        self._lambda[0] = 1 - self.R[(0, 1)] @ self.beta[1][1]

        # Ssigma는 pdf에 나온 index에서 1을 뺀 index로 사용
        self.Ssigma[0] = self.sigma_00 * self._lambda[0]
        return None

    def run(self):
        for m in range(2, self.size + 1):
            sum = self.R[(m, 1)] @ self.Phi[(m, 1)][0]
            for a in range(2, m):
                sum += self.R[(m, a)] @ self.Phi[(m, a)][0]
            self.Theta[m] = self.R[(m, m)] - sum
            self.Theta_inv[m] = np.linalg.inv(self.Theta[m])

            sum = self.R[(m, 1)] @ self.beta[m - 1][1]
            for a in range(2, m):
                sum += self.R[(m, a)] @ self.beta[m - 1][a]
            self.h[m] = self.R[(m, 0)] - sum

            self.beta[m] = np.zeros(m + 1)
            # beta_m(0)는 항상 -1
            self.beta[m][0] = -1
            self.beta[m][m] = self.Theta_inv[m] @ self.h[m]

            for a in range(1, m):
                self.beta[m][a] = self.beta[m - 1][a] - self.Phi[(m, a)][0] @ self.beta[m][m]

            
            # lambda는 pdf에 나온 index에서 2를 뺀 index로 사용
            self._lambda[m - 1] = self._lambda[m - 2] - np.reshape(self.h[m], (1, -1)) @ self.beta[m][m]
            
            # lambda는 pdf에 나온 index에서 2를 뺀 index로 사용
            # Ssigma는 pdf에 나온 index에서 1을 뺀 index로 사용
            self.Ssigma[m - 1] = self.sigma_00 * self._lambda[m - 1]

            for n in range(2, m + 1):
                sum = self.R[(n, 1) @ self.Phi[(n, 1)][m - n + 1]]
                for a in range(2, n):
                    sum += self.R[(n, a) @ self.Phi[(n, a)][m - n + 1]]

                self.Phi[(n + 1, n)][m - n] = self.Theta_inv[n] @ (self.R[(n, m + 1)] - sum)

                for a in range(1, n):
                    self.Phi[(n + 1, a)][m - n] = self.Phi[(n, a)][m - n + 1] - self.Phi[(n, a)][0] @ self.Phi[(n + 1, n)][m - n]

        return (self.beta, self.Ssigma)
