import numpy as np

def eta_lock_windows(eta_series, chern_mod2, window=5):
    locks=[]
    for i in range(len(eta_series)-window):
        e = eta_series[i:i+window]; c = chern_mod2[i:i+window]
        eta_locked = len(np.unique(np.sign(e))) == 1
        chern_locked = len(np.unique(c)) == 1
        locks.append(eta_locked and chern_locked)
    return np.array(locks, bool)

def decoder_priors_from_crossings(crossing_parity):
    priors=[]
    for p in (crossing_parity % 2):
        priors.append({'parity_flip_bias': 0.7, 'phase_flip_bias': 0.3} if p==1
                      else {'parity_flip_bias': 0.3, 'phase_flip_bias': 0.7})
    return priors
