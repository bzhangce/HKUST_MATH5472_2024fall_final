# Find the match between the EM labels and the real labels
# Let L_em[:, match_em_real_list] match L_real

# import the necessary packages
import numpy as np

def match_EM_real(L_em, L_real):
    assert L_em.shape == L_real.shape
    K = L_em.shape[1]
    dist_em_real = np.zeros((K, K))
    for ki in range(K):
        for kj in range(K):
            dist_em_real[ki, kj] = np.linalg.norm(L_real[:, ki] - L_em[:, kj], ord=1)
    match_em_real_list = np.argmin(dist_em_real, axis=1)
    if np.unique(match_em_real_list).shape[0] != K:
        print('Can not find the match')
        return
    return match_em_real_list