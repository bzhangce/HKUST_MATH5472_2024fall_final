# EM algorithm for Poisson NMF

# import the necessary packages
import numpy as np

class EM_poisson_nmf():
    def __init__(self):
        return
    
    def loglikelihood(self, X, L, F):
        return np.sum(X * np.log(L @ F.T))


    def update_params(self, X, L, F, t):
        Pi = L @ F.T
        X_div_Pi = X / Pi
        L_new =  L * (X_div_Pi @ F)
        L_new = (L_new.T / t).T
        F_new = F * (X_div_Pi.T @ L)
        F_new = F_new / np.sum(F_new, axis=0)
        return L_new, F_new
    
    def init_params_old(self, n, m, K):
        # Initialize the parameters
        # this initialization is not good
        # because it will lead to local maximum in one iteration
        # L = 1/K and Fjk = Fjt 
        L = np.ones((n, K)) / K
        F = np.ones((m, K)) / m
        return L, F
    
    def init_params_random(self, n, m, K, random_seed):
        # Initialize the parameters
        np.random.seed(random_seed)
        L = np.abs(np.random.rand(n, K))
        L = L / np.sum(L, axis=1, keepdims=True)
        F = np.abs(np.random.rand(m, K))
        F = F / np.sum(F, axis=0, keepdims=True)
        return L, F
    
    def fit(self, X, K, max_iter=300, random_seed=0):
        
        n, m = X.shape
        
        # Initialize the parameters
        L, F = self.init_params_random(n, m, K, random_seed)

        t = np.sum(X, axis=1)

        loglikelihood_list = []

        for iter_num in range(max_iter):
            L_new, F_new = self.update_params(X, L, F, t)

            L = L_new.copy()
            F = F_new.copy()

            loglikelihood = self.loglikelihood(X, L, F)
            loglikelihood_list.append(loglikelihood)

            # if iter_num % 10 == 0:
            #     print('Iter: ', iter_num)
            #     print('Loglikelihood: ', loglikelihood)
            #     print('----------------------------------')

        return L, F, loglikelihood_list


