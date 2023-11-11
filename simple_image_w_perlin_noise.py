import numpy as np
import matplotlib.pyplot as plt

# image resolution
# IPhone resolution: 2532×1170
width  = 84*6
height = 84*6

import numpy as np
import matplotlib.pyplot as plt

def perlin(x, y, seed=0):
    # permutation table
    np.random.seed(seed)
    p = np.arange(256, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    # coordinates of the top-left
    xi, yi = x.astype(int), y.astype(int)
    # internal coordinates
    xf, yf = x - xi, y - yi
    # fade factors
    u, v = fade(xf), fade(yf)
    # noise components
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    # combine noises
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)  
    return lerp(x1, x2, v) 

def lerp(a, b, x):
    "linear interpolation"
    return a + x * (b - a)

def fade(t):
    "6t^5 - 15t^4 + 10t^3"
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h, x, y):
    "grad converts h to the right gradient vector and return the dot product with (x,y)"
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y

# generating noise at multiple frequencies and adding them up
p = np.zeros((width, height))
for i in range(4):
    freq = 2**i
    lin = np.linspace(0, freq, width, endpoint=False)
    x, y = np.meshgrid(lin, lin)
    p = perlin(x, y, seed=87) / freq + p

# "shader" function
def psicadelia(img):
	for rgb_channel in range(3):
		for i in range(width):
			for j in range(height):
				img[i, j, rgb_channel] = (rgb_channel * np.cos(2*np.pi*np.cos(i/width - j*rgb_channel/width)) + np.sin(2*np.pi*np.cos(i/width - j*rgb_channel/width)))**2
    # add perlin noise on green channel
	img[:, :, 1] = np.abs(p)/np.max(np.abs(p))
	return np.clip(img, 0, 1) # safeguard

img = psicadelia(np.zeros((width, height, 3))) # rgb

# plot the image without axis or padding
plt.imshow(psicadelia(img))
plt.axis('off')
plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
plt.show()

# save image as png
plt.imsave('psicadelia.png', np.clip(img, 0, 1))

from IPython import embed; embed()
