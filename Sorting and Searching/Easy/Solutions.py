"""
35. Search Insert Position

Given a sorted array and a target value, return the index if the target is found. If not, return the index 
where it would be if it were inserted in order.

You may assume no duplicates in the array.

Example 1:
Input: [1,3,5,6], 5
Output: 2

Example 2:
Input: [1,3,5,6], 2
Output: 1

Example 3:
Input: [1,3,5,6], 7
Output: 4

Example 4:
Input: [1,3,5,6], 0
Output: 0
"""

class Solution:
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        lo = 0
        hi = len(nums) - 1
        
        while lo < hi:
            mid = (lo + hi)//2
            
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        
        return lo if target <= nums[lo] else lo + 1

"""
374. Guess Number Higher or Lower

We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I'll tell you whether the number is higher or lower.

You call a pre-defined API guess(int num) which returns 3 possible results (-1, 1, or 0):

-1 : My number is lower
 1 : My number is higher
 0 : Congrats! You got it!
 
Example :
Input: n = 10, pick = 6
Output: 6
"""
# The guess API is already defined for you.
# @param num, your guess
# @return -1 if my number is lower, 1 if my number is higher, otherwise return 0
# def guess(num):

class Solution(object):
    def guessNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        lo = 1
        hi = n
        
        while lo <= hi:
            mid = (lo + hi) // 2
            g = guess(mid)
            
            if g == 0:
                return mid
            elif g > 0:
                lo = mid + 1
            else:
                hi = mid - 1
    
"""
69. Sqrt(x)

Implement int sqrt(int x).

Compute and return the square root of x, where x is guaranteed to be a non-negative integer.

Since the return type is an integer, the decimal digits are truncated and only the integer part of the
result is returned.

Example 1:
Input: 4
Output: 2

Example 2:
Input: 8
Output: 2

Explanation: The square root of 8 is 2.82842..., and since 
             the decimal part is truncated, 2 is returned.
"""
class Solution(object):
    def mySqrt(self, x):
        lo = 1
        hi = x
        
        while lo <= hi:
            mid = (lo + hi) // 2
            sq = mid * mid
            
            if sq == x:
                return mid
            elif sq < x:
                lo = mid + 1
            else:
                hi = mid - 1
        
        return min(lo, hi)

"""
278. First Bad Version

You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest 
version of your product fails the quality check. Since each version is developed based on the previous 
version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the 
following ones to be bad.

You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a 
function to find the first bad version. You should minimize the number of calls to the API.

Example:
Given n = 5, and version = 4 is the first bad version.

call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true

Then 4 is the first bad version. 
"""

# The isBadVersion API is already defined for you.
# @param version, an integer
# @return a bool
# def isBadVersion(version):

class Solution:
    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """
        lo = 1
        hi = n

        while lo < hi:
            mid = (lo + hi)//2
            
            if isBadVersion(mid):
                hi = mid
            else:
                lo = mid + 1

        return lo
            