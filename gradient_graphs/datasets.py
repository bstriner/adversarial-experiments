import numpy as np


def polygon(k):
    angles = np.arange(k) * (2 * np.pi / k) + (np.pi / 2)
    x = np.cos(angles).reshape((-1, 1))
    y = np.sin(angles).reshape((-1, 1))
    return np.hstack((x, y))


"""
def dataset_files():
    return ["output/datasets/dataset_{}.npz".format(i) for i in range(5)]


def create_datasets():
    
    
    for path in dataset_files():
        xreal = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]).astype(np.float32)
        xfake = (np.random.random((4, 2)) * 4 - 2).astype(np.float32)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        with open(path, 'wb') as f:
            np.savez(f, xreal=xreal, xfake=xfake)
"""


def load_datasets():
    dss = []
    tri = polygon(3)
    dss.append(('triangle-inner', tri, tri * 0.5))
    dss.append(('triangle-outer', tri, tri * 1.5))

    x = [[-1, 0], [0, 0], [1, 0]]
    y = [[-.75, 0.5], [-.25, -.5], [0.75, 0]]
    dss.append(('line-1', x, y))

    y = tri * 0.75
    y[1, :] = [0.25, 0.25]
    dss.append(('triangle-offset', tri, y))
    return [(ds[0], np.array(ds[1]).astype(np.float32), np.array(ds[2]).astype(np.float32)) for ds in dss]


if __name__ == "__main__":
    load_datasets()
