import pandas as pd
import matplotlib.pyplot as plt

data = {
    'RA':       [0.1094, 0.0039, 0.0],
    'RR 0.01':  [0.2188, 0.1406, 0.1660],
    'RR 0.02':  [0.2266, 0.1484, 0.1719],
    'RR 0.03':  [0.2305, 0.1523, 0.1777],
    'RR 0.04':  [0.2344, 0.1523, 0.1777],
    'RR 0.05':  [0.2422, 0.1543, 0.1797],
    'RR 0.06':  [0.2461, 0.1582, 0.1797],
    'RR 0.07':  [0.2461, 0.1582, 0.1797],
    'RR 0.08':  [0.2480, 0.1602, 0.1797],
    'RR 0.09':  [0.2500, 0.1621, 0.1797],
    'RR 0.10':  [0.2539, 0.1660, 0.1797],
    'RR 0.11':  [0.2539, 0.1660, 0.1797],
    'RR 0.12':  [0.2578, 0.1680, 0.1797],
    'RR 0.13':  [0.2598, 0.1680, 0.1797],
    'RR 0.14':  [0.2637, 0.1719, 0.1797],
    'RR 0.15':  [0.2656, 0.1738, 0.1797],
    'RR 0.16':  [0.2656, 0.1738, 0.1797],
    'RR 0.17':  [0.2676, 0.1758, 0.1797],
    'RR 0.18':  [0.2676, 0.1758, 0.1797],
    'RR 0.19':  [0.2695, 0.1777, 0.1797],
    'RR 0.20':  [0.2695, 0.1777, 0.1797],
}

attacks = ['FGSM', 'PGD', 'CW']
df = pd.DataFrame(data, index=attacks)

# Increase figure size and reduce font
fig, ax = plt.subplots(figsize=(18, 5))
ax.axis('off')

table = ax.table(
    cellText=df.values,
    rowLabels=df.index,
    colLabels=df.columns,
    cellLoc='center',
    loc='center'
)

# Reduce font size so everything fits
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.5)

plt.tight_layout()
plt.show()
