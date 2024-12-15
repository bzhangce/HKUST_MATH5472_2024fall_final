# Generate simulated data

# import the necessary packages
import numpy as np

class Generate_Simulated_Data:
    def __init__(self):
        return

    def softmax(self, x, axis=-1):
        """
        Compute softmax values for each sets of scores in x.
        """
        assert len(x.shape) == 2
        e_x = np.exp(x - np.max(x))
        return np.exp(x - np.max(x) - np.log(e_x.sum(axis=axis, keepdims=True)))

    def mu_cov_1(self, K):
        mu1 = np.zeros(K)
        cov1 = -2 * np.ones((K, K)) + 13 * np.eye(K)
        cov1 = (cov1 + cov1.T) / 2
        return mu1, cov1
    
    def mu_cov_2(self, K):
        mu2 = np.zeros(K)
        cov2 = -2 * np.ones((K, K)) + 13 * np.eye(K)
        cov2[5, 4] = cov2[4, 5] = 8
        cov2 = (cov2 + cov2.T) / 2
        return mu2, cov2

    def generate_data(self, n, m, K, random_seed=0, choice=1):
        """
        Generate simulated data
        """
        np.random.seed(random_seed)

        if choice == 1:
            mu, cov = self.mu_cov_1(K)
        elif choice == 2:
            mu, cov = self.mu_cov_2(K)
        else:
            print("Invalid choice")
            return
        eta = np.random.multivariate_normal(mu, cov, n)

        L = self.softmax(eta)
        F = np.abs(np.random.randn(m, K))
        F = F / F.sum(axis=0, keepdims=True)

        Pi = L @ F.T

        X = np.zeros((n, m), dtype=int)
        t = np.random.poisson(1000, n)

        for i in range(n):
            X[i, :] = np.random.multinomial(t[i], Pi[i, :])
        
        return X, L, F

        