import numpy as np

def logdet(P):
    sign, ld = np.linalg.slogdet(P)
    return ld if sign > 0 else -np.inf


def pos_key(pos):
    return round(float(pos), 6)

