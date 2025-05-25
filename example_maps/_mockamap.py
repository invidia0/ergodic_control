import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv

# Load a map and dimensionally augment it
with open(os.path.join(os.path.dirname(__file__), 'complexMap.csv')) as f:
    reader = csv.reader(f)
    data = np.array(list(reader), dtype=float)

map = np.array(data)
# set to 1 the last row and column
map[-1, :] = 1
# Augment the map with a different dx
dx = 0.5
map = np.kron(map, np.ones((int(1/dx), int(1/dx))))
# Flip the map
map = np.flip(map, axis=0)
np.save(os.path.join(os.path.dirname(__file__), 'complexMap_05.npy'), map)
# Create a figure and axis
fig, ax = plt.subplots()
ax.set_aspect('equal', adjustable='box')
ax.pcolormesh(map, cmap='Greys')
plt.show()