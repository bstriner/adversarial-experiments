import numpy as np
import os


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


def load_datasets():
    x = 0.2
    dss = []
    dss.append(("square",
                [[1, 1], [1, -1], [-1, -1], [-1, 1]],
                [[x, x], [x, -x], [-x, -x], [-x, x]]))
    for path in dataset_files():
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'rb') as f:
            data = np.load(f)
            dss.append((name, data["xreal"], data["xfake"]))

    dss = [(n, np.array(xr).astype(np.float32), np.array(xf).astype(np.float32)) for n, xr, xf in dss]

    return dss


if __name__ == "__main__":
    create_datasets()
