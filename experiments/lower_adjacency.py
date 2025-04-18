import matplotlib.pyplot as plt

# Adjusted coordinates to place a1, a2, a5 at the bottom
points = {'a1': (0, 0), 'a2': (2, 0), 'a3': (1, 1.5), 'a4': (3, 1.5), 'a5': (4, 0)}
triangles = [('a1', 'a2', 'a3'), ('a2', 'a4', 'a5')]

# First plot: Highlight shared edge (Correct adjacency)
plt.figure(figsize=(5, 5))

# Plot edges first with zorder=1
for p1, p2, p3 in triangles:
    plt.plot(*zip(*[points[p1], points[p2]]), color='gray', linewidth=2, zorder=1)
    plt.plot(*zip(*[points[p2], points[p3]]), color='gray', linewidth=2, zorder=1)
    plt.plot(*zip(*[points[p3], points[p1]]), color='gray', linewidth=2, zorder=1)

# Highlight the shared edge with higher zorder
plt.plot(*zip(*[points['a1'], points['a3']]), color='blue', linewidth=3, zorder=2)
plt.plot(*zip(*[points['a3'], points['a2']]), color='orange', linewidth=3, zorder=2)

# Plot nodes with zorder=3
plt.scatter(*zip(*points.values()), color='black', s=80, zorder=3)

# Custom label positions with white background
for label, (x, y) in points.items():
    offset = -0.25 if label in ['a1', 'a2', 'a5'] else 0.2  # Increased downward offset for bottom nodes
    plt.text(x, y + offset, label,
             fontsize=12,
             ha='center',
             va='center',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.2'),
             zorder=4)

plt.xlim(-1, 5)
plt.ylim(-1.5, 3)  # Extended lower limit for label space
plt.axis('off')
plt.title("Lower Adjacency (Correct)", fontsize=14)
plt.tight_layout()
plt.show()

# Second plot: Incorrect lower adjacency
plt.figure(figsize=(5, 5))
triangles2 = [('a1', 'a2', 'a3'), ('a2', 'a5', 'a4')]

# Plot edges first
for (p1, p2, p3) in triangles2:
    color = 'blue' if (p1, p2, p3) == ('a1', 'a2', 'a3') else 'orange'
    plt.plot(*zip(*[points[p1], points[p2]]), color=color, linewidth=2, zorder=1)
    plt.plot(*zip(*[points[p2], points[p3]]), color=color, linewidth=2, zorder=1)
    plt.plot(*zip(*[points[p3], points[p1]]), color=color, linewidth=2, zorder=1)

# Plot nodes
plt.scatter(*zip(*points.values()), color='black', s=80, zorder=3)

# Labels with adjusted positions
for label, (x, y) in points.items():
    offset = -0.25 if label in ['a1', 'a2', 'a5'] else 0.2
    plt.text(x, y + offset, label,
             fontsize=12,
             ha='center',
             va='center',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.2'),
             zorder=4)

plt.xlim(-1, 5)
plt.ylim(-1.5, 3)
plt.axis('off')
plt.title("Non-Lower Adjacency (Incorrect)", fontsize=14)
plt.tight_layout()
plt.show()