import numpy as np

def manolo_colormap(t=256):
    T = np.array([[0, 0, 0],
                  [0, 0, 255],
                  [255, 0, 0],
                  [255, 255, 0],
                  [255, 255, 255]]) / 255

    x = np.array([0, 70, 130, 200, 255]) / 255

    map = np.zeros((t, 3))
    for i in range(3):
        map[:, i] = np.interp(np.linspace(0, 1, t), x, T[:, i])

    return map
