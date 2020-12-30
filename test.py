import numpy as np

a = np.ndarray((50,28,28))
b = np.moveaxis(a, 0, 2)
c = np.expand_dims(b, 0)
d = np.moveaxis(c, 0, 3)
print(d.shape)