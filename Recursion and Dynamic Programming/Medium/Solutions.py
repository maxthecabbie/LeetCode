"""
22. Generate Parentheses

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
"""

class Solution:
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = []
        self.paren_helper(n, n, "", result)
        return result
    
    def paren_helper(self, o, c, s, result):
        if o == 0 and c == 0:
            result.append(s)
        
        if o > 0:
            self.paren_helper(o-1, c, s + "(", result)
        if c > o:
            self.paren_helper(o, c-1, s + ")", result)

"""
583. Delete Operation for Two Strings

Given two words word1 and word2, find the minimum number of steps required to make word1 and word2 the same,
 where in each step you can delete one character in either string.

Example 1:
Input: "sea", "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".

Note:
The length of given words won't exceed 500.
Characters in given words can only be lower-case letters.
"""
           
class Solution:
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        dp = [[0 for _ in range(len(word2) + 1)] for _ in range(len(word1) + 1)]
        
        for i in range(1, len(dp)):
            for j in range(1, len(dp[0])):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = 1 + dp[i - 1][j - 1]
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        longest_subsequence = dp[-1][-1]
        
        return len(word1) - longest_subsequence + len(word2) - longest_subsequence

"""
62. Unique Paths

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the 
bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

Note: m and n will be at most 100.

Example 1:
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right

Example 2:
Input: m = 7, n = 3
Output: 28

https://leetcode.com/problems/unique-paths/description/
"""

class Solution:
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        moves = [[None for j in range(m)] for i in range(n)]
        return self.helper(m, n, 0, 0, moves)
    
    def helper(self, m, n, r, c, moves):
        if r >= n or c >= m:
            return 0
        
        if moves[r][c] is not None:
            return moves[r][c]
        
        if r == n - 1 and c == m - 1:
            return 1
        
        right = self.helper(m, n, r, c+1, moves)
        down = self.helper(m, n, r+1, c, moves)
        
        moves[r][c] = right + down
        return moves[r][c]

"""
64. Minimum Path Sum

Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which 
minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example:
Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
"""

class Solution:
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        cache = [[-1 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        return self.path_helper(0, 0, 0, grid, cache)
        
    
    def path_helper(self, i, j, total, grid, cache):
        if i >= len(grid) or j >= len(grid[0]):
            return sys.maxsize

        if cache[i][j] != -1:
            return cache[i][j]
        
        val = grid[i][j]
        if i == len(grid) - 1 and j == len(grid[0]) - 1:
            return total + val

        down = val + self.path_helper(i+1, j, total, grid, cache)
        right = val + self.path_helper(i, j+1, total, grid, cache)
        cache[i][j] = min(down, right)
        return cache[i][j]

"""
62. Unique Paths

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the 
bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?

https://leetcode.com/problems/unique-paths/description/
Above is a 7 x 3 grid. How many possible unique paths are there?

Note: m and n will be at most 100.

Example 1:
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right

Example 2:
Input: m = 7, n = 3
Output: 28
"""

class Solution:
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        moves = [[None for j in range(m)] for i in range(n)]
        return self.helper(m, n, 0, 0, moves)
    
    def uniquePaths_helper(self, m, n, r, c, moves):
        if r >= n or c >= m:
            return 0
        
        if moves[r][c] is not None:
            return moves[r][c]
        
        if r == n - 1 and c == m - 1:
            return 1
        
        right = self.helper(m, n, r, c+1, moves)
        down = self.helper(m, n, r+1, c, moves)
        
        moves[r][c] = right + down
        return moves[r][c]

"""
64. Minimum Path Sum

Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which 
minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example:
Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
"""

class Solution:
    def minPathSum(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        cache = [[-1 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        return self.path_helper(0, 0, 0, grid, cache)
        
    
    def path_helper(self, i, j, total, grid, cache):
        if i >= len(grid) or j >= len(grid[0]):
            return sys.maxsize
        
        val = grid[i][j]
        if i == len(grid) - 1 and j == len(grid[0]) - 1:
            return total + val
        
        if cache[i][j] != -1:
            return cache[i][j]
        
        cache[i][j] = val + min(self.path_helper(i+1, j, total, grid, cache),\
                                self.path_helper(i, j+1, total, grid, cache))
        return cache[i][j]

"""
300. Longest Increasing Subsequence

Given an unsorted array of integers, find the length of longest increasing subsequence.

Example:
Input: [10,9,2,5,3,7,101,18]
Output: 4 
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4. 

Note:
There may be more than one LIS combination, it is only necessary for you to return the length.
Your algorithm should run in O(n2) complexity.
Follow up: Could you improve it to O(n log n) time complexity?
"""

class Solution:
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 0:
            return 0
        
        lis = 1
        result = [1 for _ in range(len(nums))]
        
        for i in range(len(nums)):
            for j in range(i):
                if nums[i] > nums[j]:
                    result[i] = max(result[i], 1 + result[j])
                    lis = max(lis, result[i])
        return lis

"""
279. Perfect Squares

Given a positive integer n, find the least number of perfect square numbers (for example, 1, 4, 9, 16, ...) 
which sum to n.

Example 1:
Input: n = 12
Output: 3 
Explanation: 12 = 4 + 4 + 4.

Example 2:
Input: n = 13
Output: 2
Explanation: 13 = 4 + 9.
"""

class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        squares = self.get_squares(n)
        cache = [sys.maxsize for _ in range(n + 1)]
        cache[0] = 0
        
        for i in range(1, n+1):
            for sq in squares:
                if sq <= i:
                    cache[i] = min(cache[i], 1 + cache[i-sq])
        return cache[n]
    
    def get_squares(self, n):
        i = 1
        result = []
        
        while i*i <= n:
            result.append(i*i)
            i += 1
        
        return result

"""
139. Word Break

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s 
can be segmented into a space-separated sequence of one or more dictionary words.

Note:
The same word in the dictionary may be reused multiple times in the segmentation.
You may assume the dictionary does not contain duplicate words.

Example 1:
Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".

Example 2:
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
             Note that you are allowed to reuse a dictionary word.

Example 3:
Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: false
"""

class Solution:
    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """
        cache = {}
        return self.wordBreak_helper(0, s, wordDict, cache)
    
    def wordBreak_helper(self, start, s, wordDict, cache):
        if start == len(s):
            return True
        
        if start in cache:
            return cache[start]
        
        for word in wordDict:
            end = start + len(word)
            if word == s[start:end]:
                if self.wordBreak_helper(end, s, wordDict, cache):
                    cache[start] = True
                    return True
     
        cache[start] = False
        return False

"""
322. Coin Change

You are given coins of different denominations and a total amount of money amount. Write a function to 
compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be 
made up by any combination of the coins, return -1.

Example 1:
Input: coins = [1, 2, 5], amount = 11
Output: 3 
Explanation: 11 = 5 + 5 + 1

Example 2:
Input: coins = [2], amount = 3
Output: -1

Note:
You may assume that you have an infinite number of each kind of coin.
"""

class Solution:
    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        ways = [sys.maxsize for _ in range(amount + 1)]
        ways[0] = 0
        
        for i in range(1, len(ways)):
            for j in range(len(coins)):
                if coins[j] <= i:
                    ways[i] = min(ways[i], 1 + ways[i-coins[j]])
        
        return ways[-1] if ways[-1] < sys.maxsize else -1

"""
50. Pow(x, n)

Implement pow(x, n), which calculates x raised to the power n (xn).

Example 1:
Input: 2.00000, 10
Output: 1024.00000

Example 2:
Input: 2.10000, 3
Output: 9.26100

Example 3:
Input: 2.00000, -2
Output: 0.25000
Explanation: 2-2 = 1/22 = 1/4 = 0.25

Note:
-100.0 < x < 100.0
n is a 32-bit signed integer, within the range [−231, 231 − 1]
"""

class Solution:
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        if n < 0:
            x = 1/x
            n = -n
        
        return self.pow_helper(x, n)
    
    def pow_helper(self, x, n):
        if n == 0:
            return 1.0
        
        half = self.pow_helper(x, n//2)
        
        if n % 2 == 0:
            return half * half
        else:
            return x * half * half