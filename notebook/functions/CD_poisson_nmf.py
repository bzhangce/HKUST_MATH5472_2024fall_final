# Co-ordinate Descent algorithm for Poisson NMF

# import the necessary packages
import numpy as np

class CD_poisson_nmf():
    def __init__(self):
        return
    
    def loglikelihood(self, X, L, F):
        return np.sum(X * np.log(L @ F.T))


    def update_params(self, X, L, F, alpha):
        Pi = L @ F.T

        # update L
        G = (1 - X / Pi) @ F
        H = (X / (Pi**2)) @ (F**2)
        L_new = np.maximum(0, L - alpha * G / H)
        L_new = L_new / L_new.sum(axis=1, keepdims=True)
        # update F
        G_ = (1 - X / Pi).T @ L
        H_ = (X / (Pi**2)).T @ (L**2)
        F_new = np.maximum(0, F - alpha * G_ / H_)
        F_new = F_new / F_new.sum(axis=0, keepdims=True)

        return L_new, F_new
    
    def init_params_old(self, n, m, K):
        # Initialize the parameters
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
    
    def fit(self, X, K, max_iter=300, alpha=0.01, random_seed=0):

        n, m = X.shape
        
        # Initialize the parameters
        L, F = self.init_params_random(n, m, K, random_seed)

        loglikelihood_list = []

        for iter_num in range(max_iter):
            L_new, F_new = self.update_params(X, L, F, alpha)

            L = L_new.copy()
            F = F_new.copy()

            loglikelihood = self.loglikelihood(X, L, F)
            loglikelihood_list.append(loglikelihood)

            # if iter_num % 10 == 0:
            #     print('Iter: ', iter_num)
            #     print('Loglikelihood: ', loglikelihood)
            #     print('----------------------------------')

        return L, F, loglikelihood_list
    
            
    


