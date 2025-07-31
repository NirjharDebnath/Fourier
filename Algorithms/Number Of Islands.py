import matplotlib.pyplot as plt 
import numpy as np
import numpy as np
from noise import pnoise2
import sys
sys.setrecursionlimit(100000)


class Solution:
    def countIslands(self, grid):
        n = len(grid)
        m = len(grid[0])
        visited = [[0] * m for _ in range(n)]

        def dfs(i, j):
            visited[i][j] = 1
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    row, col = i + dr, j + dc
                    if 0 <= row < n and 0 <= col < m:
                        if grid[row][col] == 1 and not visited[row][col]:
                            dfs(row, col)

        count = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1 and not visited[i][j]:
                    count += 1
                    dfs(i, j)

        return count

m, n = 100, 100
  # Requires `noise` package: pip install noise

def generate_perlin_map(width, height, scale=15, threshold=0.1):
    world = [[0 for _ in range(width)] for _ in range(height)]
    for i in range(height):
        for j in range(width):
            noise_val = pnoise2(i / scale, j / scale, octaves=6)
            if noise_val > threshold:
                world[i][j] = 1  # Land
            else:
                world[i][j] = 0  # Water
    return world

# Example usage:
map1 = generate_perlin_map(100, 100, scale=20, threshold=0.05)


findilse = Solution()
num_ilse = findilse.countIslands(map1)
print("Number of islands:", num_ilse)

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["#3a93ce", "#48d33e"])  # blue, green

# Plot
plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(map1, cmap=cmap)
plt.title(f"Perlin Archipelago — Islands Found: {num_ilse}", fontsize=14)
plt.axis('off')
plt.show()
