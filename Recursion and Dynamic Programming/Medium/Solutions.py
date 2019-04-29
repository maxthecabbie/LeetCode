"""
931. Minimum Falling Path Sum

Given a square array of integers A, we want the minimum sum of a falling path through A.

A falling path starts at any element in the first row, and chooses one element from each row. The next row's
choice must be in a column that is different from the previous row's column by at most one.

Example 1:
Input: [[1,2,3],[4,5,6],[7,8,9]]
Output: 12

Explanation: 
The possible falling paths are:
[1,4,7], [1,4,8], [1,5,7], [1,5,8], [1,5,9]
[2,4,7], [2,4,8], [2,5,7], [2,5,8], [2,5,9], [2,6,8], [2,6,9]
[3,5,7], [3,5,8], [3,5,9], [3,6,8], [3,6,9]
The falling path with the smallest sum is [1,4,7], so the answer is 12.

Note:
1 <= A.length == A[0].length <= 100
-100 <= A[i][j] <= 100
"""

class Solution(object):
    def minFallingPathSum(self, A):
        """
        :type A: List[List[int]]
        :rtype: int
        """
        min_falling_path = sys.maxsize
        
        cache = [row[:] for row in A]
        
        for i in range(1, len(cache)):
            for j in range(len(cache[0])):
                
                min_path = sys.maxsize
                
                for r, c in ((i-1, j-1), (i-1, j), (i-1, j+1)):                 
                    if 0 <= c < len(cache[0]):
                        min_path = min(min_path, cache[r][c])
             
                cache[i][j] += min_path
        
        for n in cache[-1]:
            min_falling_path = min(min_falling_path, n)
        
        return min_falling_path

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
638. Shopping Offers

In LeetCode Store, there are some kinds of items to sell. Each item has a price.

However, there are some special offers, and a special offer consists of one or more different kinds of items 
with a sale price.

You are given the each item's price, a set of special offers, and the number we need to buy for each item. 
The job is to output the lowest price you have to pay for exactly certain items as given, where you could 
make optimal use of the special offers.

Each special offer is represented in the form of an array, the last number represents the price you need to 
pay for this special offer, other numbers represents how many specific items you could get if you buy this 
offer.

You could use any of special offers as many times as you want.

Example 1:
Input: [2,5], [[3,0,5],[1,2,10]], [3,2]
Output: 14
Explanation: 
There are two kinds of items, A and B. Their prices are $2 and $5 respectively. 
In special offer 1, you can pay $5 for 3A and 0B
In special offer 2, you can pay $10 for 1A and 2B. 
You need to buy 3A and 2B, so you may pay $10 for 1A and 2B (special offer #2), and $4 for 2A.

Example 2:
Input: [2,3,4], [[1,1,0,4],[2,2,1,9]], [1,2,1]
Output: 11
Explanation: 
The price of A is $2, and $3 for B, $4 for C. 
You may pay $4 for 1A and 1B, and $9 for 2A ,2B and 1C. 
You need to buy 1A ,2B and 1C, so you may pay $4 for 1A and 1B (special offer #1), and $3 for 1B, $4 for 1C. 
You cannot add more items, though only $9 for 2A ,2B and 1C.

Note:
There are at most 6 kinds of items, 100 special offers.
For each item, you need to buy at most 6 of them.
You are not allowed to buy more items than you want, even if that would lower the overall price.
"""

class Solution:
    def shoppingOffers(self, price, special, needs):
        """
        :type price: List[int]
        :type special: List[List[int]]
        :type needs: List[int]
        :rtype: int
        """
        return self.shop_helper(price, special, needs, {})
        
    def shop_helper(self, price, special, needs, cache):
        if sum(needs) == 0:
            return 0
        
        key = str(needs)
        if cache.get(key, None):
            return cache[key]
        
        new_needs = [n-1 if n > 0 else n for n in needs]
        cost = sum([price[i] for i in range(len(needs)) if needs[i] > 0])
        cache[key] = cost + self.shop_helper(price, special, new_needs, cache)
        
        for s in special:
            if self.can_special(s, needs):
                new_needs = [needs[i] - s[i] for i in range(len(needs))]
                cost = s[-1] + self.shop_helper(price, special, new_needs, cache)
                cache[key] = min(cache[key], cost)
                
        return cache[key]
        
    def can_special(self, special, needs):
        for i in range(len(needs)):
            if needs[i] < special[i]:
                return False
        return True

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
        return self.path_helper(0, 0, grid, cache)
        
    
    def path_helper(self, i, j, grid, cache):
        if i >= len(grid) or j >= len(grid[0]):
            return sys.maxsize
        
        val = grid[i][j]
        if i == len(grid) - 1 and j == len(grid[0]) - 1:
            return val
        
        if cache[i][j] != -1:
            return cache[i][j]
        
        cache[i][j] = val + min(self.path_helper(i+1, j, grid, cache),\
                                self.path_helper(i, j+1, grid, cache))
        return cache[i][j]

"""
247. Strobogrammatic Number II

A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).

Find all strobogrammatic numbers that are of length = n.

Example:
Input:  n = 2
Output: ["11","69","88","96"]
"""

class Solution:
    def findStrobogrammatic(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        return self.helper(n, True)

    def helper(self, n, first):
        if n == 0:
            return [""]
        if n == 1:
            return ["1", "8", "0"]
        
        mid = self.helper(n-2, False)
        result = []
        
        for n in mid:
            if not first:
                result.append("0" + n + "0")
            result.append("1" + n + "1")
            result.append("6" + n + "9")
            result.append("8" + n + "8")
            result.append("9" + n + "6")
        
        return result

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
        cache = [sys.maxsize for _ in range(n+1)]
        cache[0] = 0
        
        for i in range(len(cache)):
            j = 1
            while j*j <= i:
                cache[i] = min(cache[i], 1 + cache[i - j*j])
                j += 1
        
        return cache[-1]

"""
935. Knight Dialer

A chess knight can move as indicated in the chess diagram below:
https://leetcode.com/problems/knight-dialer/

This time, we place our chess knight on any numbered key of a phone pad (indicated above), and the knight
makes N-1 hops.  Each hop must be from one key to another numbered key.

Each time it lands on a key (including the initial placement of the knight), it presses the number of that
key, pressing N digits total.

How many distinct numbers can you dial in this manner?

Since the answer may be large, output the answer modulo 10^9 + 7.

Example 1:
Input: 1
Output: 10

Example 2:
Input: 2
Output: 20

Example 3:
Input: 3
Output: 46
"""

class Solution(object):
    def knightDialer(self, N):
        """
        :type N: int
        :rtype: int
        """
        num_mapping = {
            0: [4, 6], 1: [8, 6], 2: [7, 9],
            3: [4, 8], 4: [3, 9, 0], 5: [],
            6: [1, 7, 0], 7: [2, 6], 8: [1, 3],
            9: [2, 4]
        }
        
        counts = {n:1 for n in range(0, 10)}
        
        for _ in range(N-1):
            new_count = {n:0 for n in range(0, 10)}
            
            for n in counts:
                for next_num in num_mapping[n]:
                    new_count[next_num] += counts[n]
            
            counts = new_count
        
        return sum([counts[key] for key in counts]) % (10**9 + 7)
        

"""
213. House Robber II

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money
stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the
last one. Meanwhile, adjacent houses have security system connected and it will automatically contact the 
police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum
amount of money you can rob tonight without alerting the police.

Example 1:
Input: [2,3,2]
Output: 3

Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent
houses.
             
Example 2:
Input: [1,2,3,1]
Output: 4

Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3). Total amount you can rob = 1 + 3 = 4.
"""
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 1:
            return nums[0]
        return max(self.rob_impl(nums[1:]), self.rob_impl(nums[:-1]))
    
    def rob_impl(self, nums):
        max_money = 0
        prev = two_away = 0
        
        for i in range(len(nums)):
            max_money = max(prev, two_away + nums[i])
            two_away = prev
            prev = max_money
        
        return max_money

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
221. Maximal Square

Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its
area.

Example:
Input: 
1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0
Output: 4
"""

class Solution(object):
    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        max_side_len = 0
        
        if not matrix:
            return max_side_len
        
        m, n = len(matrix), len(matrix[0])
        cache = [[0 for _ in range(n+1)] for _ in range(m+1)]
        
        for i in range(1, len(cache)):
            for j in range(1, len(cache[0])):
                if matrix[i-1][j-1] == "1":
                    cache[i][j] = 1 + min(cache[i][j-1],
                                          min(cache[i-1][j-1], cache[i-1][j]))
                    max_side_len = max(max_side_len, cache[i][j])
        
        return max_side_len * max_side_len
        
        

"""
55. Jump Game

Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

Example 1:
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.

Example 2:
Input: [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum
jump length is 0, which makes it impossible to reach the last index.
"""

class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        cache = [None for _ in range(len(nums))]
        return self.can_jump_impl(0, nums, cache)
    
    def can_jump_impl(self, i, nums, cache):
        if i == len(nums) - 1:
            return True
        
        if cache[i] is not None:
            return False
        
        for j in range(1, nums[i] + 1):
            if i + j <= len(nums) - 1:
                if self.can_jump_impl(i+j, nums, cache):
                    return True
        
        cache[i] = False
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

"""
91. Decode Ways

A message containing letters from A-Z is being encoded to numbers using the following mapping:

'A' -> 1
'B' -> 2
...
'Z' -> 26

Given a non-empty string containing only digits, determine the total number of ways to decode it.

Example 1:
Input: "12"
Output: 2
Explanation: It could be decoded as "AB" (1 2) or "L" (12).

Example 2:
Input: "226"
Output: 3
Explanation: It could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).
"""

class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        cache = {}
        return self.num_decodings_impl(0, s, cache)

    def num_decodings_impl(self, i, s, cache):
        if i == len(s):
            return 1
        elif s[i] == "0":
            return 0
        elif s[i:] in cache:
            return cache[s[i:]]

        ways = self.num_decodings_impl(i+1, s, cache)

        if i+1 < len(s) and s[i : i+2] <= "26":
            ways += self.num_decodings_impl(i+2, s, cache)
        
        cache[s[i:]] = ways
        return ways