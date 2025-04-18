import matplotlib.pyplot as plt

# Original points and triangles
points = {'a1': (0, 0), 'a2': (2, 0), 'a3': (1, 1.5), 'a4': (3, 1.5), 'a5': (4, 0)}
triangles = [('a1', 'a2', 'a3'), ('a2', 'a4', 'a5')]

# First plot: Upper adjacency example
plt.figure(figsize=(5, 5))

# Plot the original triangles
for p1, p2, p3 in triangles:
    plt.plot(*zip(*[points[p1], points[p2]]), color='gray', linewidth=2)
    plt.plot(*zip(*[points[p2], points[p3]]), color='gray', linewidth=2)
    plt.plot(*zip(*[points[p3], points[p1]]), color='gray', linewidth=2)

# Highlight triangle [a1, a2, a3] with a single color
plt.plot(*zip(*[points['a1'], points['a2']]), color='blue', linewidth=3)
plt.plot(*zip(*[points['a2'], points['a3']]), color='blue', linewidth=3)
plt.plot(*zip(*[points['a3'], points['a1']]), color='blue', linewidth=3)

# Plot nodes
plt.scatter(*zip(*points.values()), color='black', s=80)
for label, (x, y) in points.items():
    offset = -0.25 if label in ['a1', 'a2', 'a5'] else 0.2
    plt.text(x, y + offset, label,
             fontsize=12,
             ha='center',
             va='center',
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle='round,pad=0.2'))

plt.xlim(-1, 5)
plt.ylim(-1.5, 3)
plt.axis('off')
plt.title("Upper Adjacency (Highlighted Triangle)", fontsize=14)
plt.tight_layout()
plt.show()