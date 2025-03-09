import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ergodic_control import models, utilities

# Setup matplotlib properties, remove type 3 fonts
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
# Use tex
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

# Load the data
hedac_r1 = []
hedac_r4 = []
smc_r1 = []
smc_r4 = []
proposed_r1 = []
proposed_r4 = []
voronoi = []

nbSims = 3
# Load the data
for i in [1, 4]:
    for dir in ['hedac', 'smc', 'proposed', 'voronoi']:
        if dir == 'voronoi':
            for j in [1, 2, 3]:
                path = os.path.join(os.path.dirname(__file__), f'{dir}_sim{j}/ergodic_metrics.npy')
                data = np.load(path)
                voronoi.append(data)
        else:
            for j in [1, 2, 3]:
                path = os.path.join(os.path.dirname(__file__), f'{dir}_sim{j}_r{i}/ergodic_metrics.npy')
                data = np.load(path)
                if dir == 'hedac':
                    if i == 1:
                        hedac_r1.append(data)
                    else:
                        hedac_r4.append(data)
                elif dir == 'smc':
                    if i == 1:
                        smc_r1.append(data)
                    else:
                        smc_r4.append(data)
                else:
                    if i == 1:
                        proposed_r1.append(data)
                    else:
                        proposed_r4.append(data)
    

nbDataPoints = 8000

hedac_1_mean = np.mean(hedac_r1, axis=0).flatten()
hedac_1_std = np.std(hedac_r1, axis=0).flatten()

hedac_4_mean = np.mean(hedac_r4, axis=0).flatten()
hedac_4_std = np.std(hedac_r4, axis=0).flatten()

smc_1_mean = np.mean(smc_r1, axis=0).flatten()
smc_1_std = np.std(smc_r1, axis=0).flatten()

smc_4_mean = np.mean(smc_r4, axis=0).flatten()
smc_4_std = np.std(smc_r4, axis=0).flatten()

voronoi_means = []
voronoi_stds = []
for data in voronoi:
    voronoi_means.append(np.mean(data, axis=0))
    voronoi_stds.append(np.std(data, axis=0))

voronoi_mean = np.mean(np.array(voronoi_means), axis=0).flatten()
voronoi_std = np.mean(np.array(voronoi_stds), axis=0).flatten()

proposed_1_mean = np.mean(proposed_r1, axis=0).flatten()
proposed_1_std = np.std(proposed_r1, axis=0).flatten()

proposed_4_means = []
proposed_4_stds = []
for data in proposed_r4:
    proposed_4_means.append(np.mean(data, axis=1))
    proposed_4_stds.append(np.std(data, axis=1))

proposed_4_mean = np.mean(np.array(proposed_4_means), axis=0).flatten()
proposed_4_std = np.mean(np.array(proposed_4_stds), axis=0).flatten()

plot_means = [hedac_1_mean,
                hedac_4_mean,
                smc_1_mean,
                smc_4_mean,
                voronoi_mean,
                proposed_1_mean,
                proposed_4_mean]

plot_stds = [hedac_1_std,
                hedac_4_std,
                smc_1_std,
                smc_4_std,
                voronoi_std,
                proposed_1_std,
                proposed_4_std]

# Plot the data
width = 7.16 # Full page width for double-column papers
golden_ratio = 0.618 # Golden ratio
height = width * golden_ratio
fontdict = {'weight': 'bold', 'size': 14, 'family': 'serif'}

# fig = plt.figure(figsize=(width, height))
fig = plt.figure()
ax = fig.add_subplot(111)

fontsize = 16
ax.set_xlabel(r'\textbf{Time Steps}', fontsize=fontsize)
ax.set_ylabel(r'\fontfamily{cmr}\textbf{Performance} $\mathcal{H}(\mathbf{x},t)$', fontsize=fontsize)

# Set a title for the plot
ax.set_title(r'\textbf{Performance Comparison on Simple Environment}', fontsize=fontsize)

# Use gridlines for better readability
ax.grid(True, linestyle='--', alpha=0.5)

# Set the interval for markers
marker_interval = 250  # Show markers every 10 steps
alpha = 0.1  # Set the transparency of the error bands

color = plt.cm.brg(np.linspace(0, 1, 6))
markers = ['o', 's', '^', 'X', 'D', 'v', 'P']
labels = ['HEDAC [7] - 1 Robot', 'HEDAC [7] - 4 Robots', 'SMC [15]- 1 Robot', 'SMC [1] (Only Consensus) - 4 Robots', 
          'Static Voronoi [14] - 8 Robots',
          'Proposed - 1 Robot', 'Proposed - 4 Robots']

timestep = 5000
# Cut to the same length
for i, data in enumerate(plot_means):
    plot_means[i] = data[:timestep]
    plot_stds[i] = plot_stds[i][:timestep]

for i, data in enumerate(plot_means):
    ax.plot(data, label=labels[i], linestyle='-', marker=markers[i], markersize=7, markevery=marker_interval, linewidth=2)
    ax.fill_between(range(timestep), data - plot_stds[i], data + plot_stds[i], alpha=alpha)

# Adjust the legend to avoid overlap and increase font size
plt.legend(loc='upper center', fontsize=13, frameon=True, shadow=True, fancybox=True, ncol=2,
            bbox_to_anchor=(0.5, -0.15))
        #    bbox_to_anchor=(1, -0.5))
plt.savefig('performance_simple.pdf', format='pdf', dpi=600, bbox_inches='tight')
# Show the plot
plt.tight_layout()  # Ensures that everything fits into the figure area
plt.show()
