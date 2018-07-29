"""
280. Wiggle Sort

Given an unsorted array nums, reorder it in-place such that nums[0] <= nums[1] >= nums[2] <= nums[3]....

Example:
Input: nums = [3,5,2,1,6,4]
Output: One possible answer is [3,5,1,6,2,4]

Solution Explanation
- Consider three consecutive indices in nums, n-1, n, and n+1. If n-1 is greater than n, we swap them and 
then if n+1 is greater than n we swap them
- If we now move up two numbers from n to n+2, we can repeat the process. n+1 will be the number to the left 
of n+2 and will only be swapped with a lower number, so the property of the middle number being the highest 
always holds
"""

class Solution:
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        i = 1
        
        while i < len(nums):
            if nums[i-1] > nums[i]:
                nums[i-1], nums[i] = nums[i], nums[i-1]
            if i + 1 < len(nums) and nums[i+1] > nums[i]:
                nums[i+1], nums[i] = nums[i], nums[i+1]
            i += 2

"""
347. Top K Frequent Elements

Given a non-empty array of integers, return the k most frequent elements.

For example,
Given [1,1,1,2,2,3] and k = 2, return [1,2].

Note: 
You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Your algorithm's time complexity must be better than O(n log n), where n is the array's size.
"""

class Solution:
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        freq = {}
        buckets = [[] for _ in range(len(nums) + 1)]
        
        for n in nums:
            freq[n] = freq.get(n, 0) + 1
        
        for key in freq:
            buckets[freq[key]].append(key)
        
        result = []
        for i in range(len(buckets) - 1, -1, -1):
            for n in buckets[i]:
                if k == 0:
                    break
                result.append(n)
                k -= 1
        
        return result

"""
524. Longest Word in Dictionary through Deleting

Given a string and a string dictionary, find the longest string in the dictionary that can be formed by 
deleting some characters of the given string. If there are more than one possible results, return the 
longest word with the smallest lexicographical order. If there is no possible result, return the empty 
string.

Example 1:
Input:
s = "abpcplea", d = ["ale","apple","monkey","plea"]

Output: 
"apple"

Example 2:
Input:
s = "abpcplea", d = ["a","b","c"]

Output: 
"a"

Note:
All the strings in the input will only contain lower-case letters.
The size of the dictionary won't exceed 1,000.
The length of all the strings in the input won't exceed 1,000.
"""

class Solution:
    def findLongestWord(self, s, d):
        """
        :type s: str
        :type d: List[str]
        :rtype: str
        """
        d.sort(key = lambda s: (-len(s), s))
        
        for word in d:
            i = 0
            for ch in s:
                if ch == word[i]:
                    i += 1
                if i == len(word):
                    return word
        return ""

"""
75. Sort Colors

Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same 
color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note: You are not suppose to use the library's sort function for this problem.

Example:
Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]

Follow up:
A rather straight forward solution is a two-pass algorithm using counting sort. First, iterate the array 
counting number of 0's, 1's, and 2's, then overwrite array with total number of 0's, then 1's and followed 
by 2's.

Could you come up with a one-pass algorithm using only constant space?
"""

class Solution:
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        r = w = b = 0
        
        for n in nums:
            if n == 0:
                r += 1
            elif n == 1:
                w += 1
            else:
                b += 1
        
        for i in range(len(nums)):
            if r > 0:
                nums[i] = 0
                r -= 1
            elif w > 0:
                nums[i] = 1
                w -= 1
            else:
                nums[i] = 2

# Alternative single pass solution
class Solution:
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        i = j = 0
        k = len(nums) - 1
        
        while i <= k:
            if nums[i] == 0:
                nums[i], nums[j] = nums[j], nums[i]
                j += 1
            elif nums[i] == 2:
                nums[i], nums[k] = nums[k], nums[i]
                i -= 1
                k -= 1
            i += 1

"""
162. Find Peak Element

A peak element is an element that is greater than its neighbors.

Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that nums[-1] = nums[n] = -∞.

Example 1:
Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.

Example 2:
Input: nums = [1,2,1,3,5,6,4]
Output: 1 or 5 
Explanation: Your function can return either index number 1 where the peak element is 2, 
             or index number 5 where the peak element is 6.

Note:
Your solution should be in logarithmic complexity.
"""

class Solution:
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        lo = 0
        hi = len(nums) - 1
        
        while lo < hi:
            mid = (lo + hi)//2
            if nums[mid] < nums[mid+1]:
                lo = mid + 1
            else:
                hi = mid
        
        return lo

"""
240. Search a 2D Matrix II

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following 
properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.

Example:
Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]

Given target = 5, return true.

Given target = 20, return false.
"""

class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        i = len(matrix) - 1
        j = 0
        
        while i >= 0 and j < len(matrix[0]):
            print(i, j)
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                i -= 1
            else:
                j += 1
        
        return False

"""
33. Search in Rotated Sorted Array

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

Example 2:
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
"""

class Solution:
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        lo = 0
        hi = len(nums) - 1
        
        while lo <= hi:
            mid = (lo + hi)//2
            
            if nums[mid] == target:
                return mid
            elif self.is_between(nums[lo], nums[mid], target):
                hi = mid - 1
            else:
                lo = mid + 1
        
        return -1
    
    def is_between(self, lo, hi, n):
        if lo <= hi:
            return n >= lo and n <= hi
        else:
            return n >= lo or n <= hi

class Solution:
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        left = self.search_border(nums, target, True)
        if len(nums) <= 0 or nums[left] != target:
            return [-1, -1]
        
        right = self.search_border(nums, target, False)
        if nums[right] != target:
            right -= 1
            
        return [left, right]
    
    def search_border(self, nums, target, left):
        lo = 0
        hi = len(nums) - 1
        
        while lo < hi:
            mid = (lo + hi)//2
            if nums[mid] == target:
                if left:
                    hi = mid
                else:
                    lo = mid + 1
            elif nums[mid] > target:
                hi = mid - 1
            else:
                lo = mid + 1
        
        return lo

"""
34. Find First and Last Position of Element in Sorted Array

Given an array of integers nums sorted in ascending order, find the starting and ending position of a given 
target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

Example 1:
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]

Example 2:
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
"""

class Solution:
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        left = self.search_border(nums, target, True)
        if len(nums) <= 0 or nums[left] != target:
            return [-1, -1]
        
        right = self.search_border(nums, target, False)
        if nums[right] != target:
            right -= 1
            
        return [left, right]
    
    def search_border(self, nums, target, left):
        lo = 0
        hi = len(nums) - 1
        
        while lo < hi:
            mid = (lo + hi)//2
            if nums[mid] == target and left:
                hi = mid
            elif nums[mid] > target:
                hi = mid - 1
            else:
                lo = mid + 1
        
        return lo