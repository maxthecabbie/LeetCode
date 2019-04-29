"""
63. Unique Paths II

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the
bottom-right corner of the grid (marked 'Finish' in the diagram below).

Now consider if some obstacles are added to the grids. How many unique paths would there be?

https://leetcode.com/problems/unique-paths-ii/

An obstacle and empty space is marked as 1 and 0 respectively in the grid.

Note: m and n will be at most 100.

Example 1:
Input:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
Output: 2

Explanation:
There is one obstacle in the middle of the 3x3 grid above.
There are two ways to reach the bottom-right corner:
1. Right -> Right -> Down -> Down
2. Down -> Down -> Right -> Right
"""

class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        """
        :type obstacleGrid: List[List[int]]
        :rtype: int
        """
        if not obstacleGrid:
            return 0
        
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        cache = [[-1 for _ in range(n)] for _ in range(m)]
        return self.unique_paths_impl(0, 0, obstacleGrid, cache)
    
    def unique_paths_impl(self, i, j, grid, cache):
        if 0 <= i < len(grid) and 0 <= j < len(grid[0]):
            
            if grid[i][j] == 1:
                return 0
            elif i == len(grid) - 1 and j == len(grid[0]) - 1:
                return 1
            elif cache[i][j] != -1:
                return cache[i][j]
            
            cache[i][j] = self.unique_paths_impl(i, j+1, grid, cache) +\
            self.unique_paths_impl(i+1, j, grid, cache)
            
            return cache[i][j]
        
        return 0