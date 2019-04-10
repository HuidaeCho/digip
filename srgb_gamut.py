#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# http://www.ryanjuckett.com/programming/rgb-color-space-conversion/
rxy = [0.64, 0.33]
gxy = [0.30, 0.60]
bxy = [0.15, 0.06]
wxy = [0.3127, 0.3290]

xx = []
yy = []
c = []

delta = 0.01
for X in np.arange(4.2e-05, 1.0622, delta):
    for Y in np.arange(1.5e-05, 1.0, delta):
        for Z in np.arange(0.0, 1.7826, delta):
            XYZ = X + Y + Z
            x = X / XYZ
            y = Y / XYZ
            z = Z / XYZ
            r = 3.24103 * X - 1.53741 * Y - 0.49862 * Z
            g = -0.969242 * X + 1.87596 * Y + 0.041555 * Z
            b = 0.055632 * X - 0.203979 * Y + 1.05698 * Z
            r = 12.92 * r if r <= 0.0031308 else 1.055 * r**(1/2.4) - 0.055
            g = 12.92 * g if g <= 0.0031308 else 1.055 * g**(1/2.4) - 0.055
            b = 12.92 * b if b <= 0.0031308 else 1.055 * b**(1/2.4) - 0.055
            if r >= 0 and r <= 1 and g >= 0 and g <= 1 and b >= 0 and b <= 1:
                xx.append(x)
                yy.append(y)
                R = int(255 * r)
                G = int(255 * g)
                B = int(255 * b)
                c.append('#{:02x}{:02x}{:02x}'.format(R, G, B))

plt.figure(figsize=(8, 8))
plt.title('sRGB Gamut')
plt.xlim(0, np.max(yy)+0.15)
plt.ylim(0, np.max(yy)+0.15)
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(xx, yy, c=c)
plt.savefig('srgb_gamut.png')
plt.show()
