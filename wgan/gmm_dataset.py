import numpy as np

def gmm_dataset(k, n, sigma = 1e-1):
    assert(n%k == 0)
    nk = n/k
    angles = np.arange(k)*2*np.pi/k
    positions = np.hstack((np.cos(angles).reshape((-1,1)), np.sin(angles).reshape((-1,1))))
    data = []
    for i in range(k):
        noise = np.random.normal(0, sigma, (nk, 2))
        datum = positions[i,:]+noise
        data.append(datum)
    dataset = np.vstack(data)
    np.random.shuffle(dataset)
    return dataset
