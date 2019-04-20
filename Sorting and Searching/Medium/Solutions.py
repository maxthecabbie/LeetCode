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

# Alternative solution using bucket sort
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        res = []
        counts = collections.Counter(nums)
        counts_arr = [[counts[n], n] for n in counts]
        self.bucket_sort(counts_arr)
        
        for i in range(k):
            res.append(counts_arr[~i][1])
        
        return res
    
    def bucket_sort(self, arr):
        if not arr:
            return arr
        
        min_n = max_n = arr[0][0]
        
        for entry in arr:
            min_n = min(min_n, entry[0])
            max_n = max(max_n, entry[0])
        
        buckets = [[] for _ in range((max_n - min_n) + 1)]
        
        for entry in arr:
            buckets[entry[0] - min_n].append(entry)
        
        i = 0
        for b in buckets:
            for entry in b:
                arr[i] = entry
                i += 1

"""
911. Online Election

In an election, the i-th vote was cast for persons[i] at time times[i].

Now, we would like to implement the following query function: TopVotedCandidate.q(int t) will return the
number of the person that was leading the election at time t.  

Votes cast at time t will count towards our query.  In the case of a tie, the most recent vote (among tied
candidates) wins.

Example 1:
Input: ["TopVotedCandidate","q","q","q","q","q","q"], [[[0,1,1,0,0,1,0],[0,5,10,15,20,25,30]],[3],[12],[25],
[15],[24],[8]]
Output: [null,0,1,1,0,0,1]

Explanation: 
At time 3, the votes are [0], and 0 is leading.
At time 12, the votes are [0,1,1], and 1 is leading.
At time 25, the votes are [0,1,1,0,0,1], and 1 is leading (as ties go to the most recent vote.)
This continues for 3 more queries at time 15, 24, and 8.
 
Note:
1 <= persons.length = times.length <= 5000
0 <= persons[i] <= persons.length
times is a strictly increasing array with all elements in [0, 10^9].
TopVotedCandidate.q is called at most 10000 times per test case.
TopVotedCandidate.q(int t) is always called with t >= times[0].
"""

class TopVotedCandidate(object):
    def __init__(self, persons, times):
        """
        :type persons: List[int]
        :type times: List[int]
        """
        self.leads = [persons[0]] if persons else []
        self.times = times
        counts = collections.defaultdict(int)
        
        if persons:
            counts[persons[0]] += 1
        
        for i in range(1, len(persons)):
            p = persons[i]
            counts[p] += 1
            
            lead = self.leads[i-1]
            if counts[p] >= counts[lead]:
                lead = p

            self.leads.append(lead)
        

    def q(self, t):
        """
        :type t: int
        :rtype: int
        """
        lead_idx = self.search(t)
        return self.leads[lead_idx]
    
    def search(self, t):
        lo, hi = 0, len(self.times) - 1
        result = -1
        
        while lo <= hi:
            mid = (lo + hi) // 2
            
            if t >= self.times[mid]:
                result = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        return result
# Your TopVotedCandidate object will be instantiated and called as such:
# obj = TopVotedCandidate(persons, times)
# param_1 = obj.q(t)

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
528. Random Pick with Weight

Given an array w of positive integers, where w[i] describes the weight of index i, write a function 
pickIndex which randomly picks an index in proportion to its weight.

Note:
1 <= w.length <= 10000
1 <= w[i] <= 10^5
pickIndex will be called at most 10000 times.

Example 1:
Input: 
["Solution","pickIndex"]
[[[1]],[]]
Output: [null,0]

Example 2:
Input: 
["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
[[[1,3]],[],[],[],[],[]]
Output: [null,0,1,1,1,0]

Explanation of Input Syntax:
The input is two lists: the subroutines called and their arguments. Solution's constructor has one argument, 
the array w. pickIndex has no arguments. Arguments are always wrapped with a list, even if there aren't any.
"""

class Solution:

    def __init__(self, w):
        """
        :type w: List[int]
        """
        self.choices = []
        self.tot = 0
        
        last = 0
        for i in range(len(w)):
            self.tot += w[i]
            self.choices.append((last + 1, w[i] + last))
            last += w[i]
            
    def pickIndex(self):
        """
        :rtype: int
        """
        t = random.randrange(1, self.tot + 1)
        lo, hi = 0, len(self.choices) - 1

        while lo < hi:
            mid = (lo + hi)//2
            r = self.choices[mid]
            
            if t >= r[0] and t <= r[1]:
                return mid
            elif t < r[0]:
                hi = mid - 1
            else:
                lo = mid + 1
                
        return lo

# Your Solution object will be instantiated and called as such:
# obj = Solution(w)
# param_1 = obj.pickIndex()

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
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                i -= 1
            else:
                j += 1
        
        return False

"""
702. Search in a Sorted Array of Unknown Size

Given an integer array sorted in ascending order, write a function to search target in nums.  If target 
exists, then return its index, otherwise return -1. However, the array size is unknown to you. You may only 
access the array using an ArrayReader interface, where ArrayReader.get(k) returns the element of the array 
at index k (0-indexed).

You may assume all integers in the array are less than 10000, and if you access the array out of bounds, 
ArrayReader.get will return 2147483647.
 
Example 1:
Input: array = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4

Example 2:
Input: array = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
 

Note:
You may assume that all elements in the array are unique.
The value of each element in the array will be in the range [-9999, 9999].
"""

class Solution:
    def search(self, reader, target):
        """
        :type reader: ArrayReader
        :type target: int
        :rtype: int
        """
        lo = 0
        hi = 1
        while reader.get(hi) < target:
            hi *= 2
        
        while lo <= hi:
            mid = (lo + hi)//2
            val = reader.get(mid)
            
            if val == target:
                return mid
            elif val > target:
                hi = mid - 1
            else:
                lo = mid + 1
                
        return -1

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
        return n >= lo or n <= hi

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

# Alternative solution that is much easier to understand
class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        
        left = self.search(nums, target, True)
        right = self.search(nums, target, False)
        return [left, right]
    
    def search(self, nums, target, left):
        lo, hi = 0, len(nums) - 1
        result = -1
        
        while lo <= hi:
            mid = (lo + hi) // 2
            
            if nums[mid] == target:
                result = mid
                if left:
                    hi = mid - 1
                else:
                    lo = mid + 1
            elif nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        
        return result

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

class Solution(object):
    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        sorted_nums = sorted(nums)
        curr = len(sorted_nums) - 1
        
        for i in range(1, len(nums), 2):
            nums[i] = sorted_nums[curr]
            curr -= 1
        
        for i in range(0, len(nums), 2):
            nums[i] = sorted_nums[curr]
            curr -= 1

"""
179. Largest Number

Given a list of non negative integers, arrange them such that they form the largest number.

Example 1:
Input: [10,2]
Output: "210"

Example 2:
Input: [3,30,34,5,9]
Output: "9534330"

Note: The result may be very large, so you need to return a string instead of an integer.
"""

class Solution(object):
    def largestNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: str
        """
        nums = [str(n) for n in nums]
        nums.sort(self.cmp_nums)
        res = "".join(nums)
        return res if not res.startswith("0") else "0"
    
    def cmp_nums(self, n1, n2):
        concat1 = n1 + n2
        concat2 = n2 + n1
        
        if concat1 > concat2:
            return -1
        elif concat1 < concat2:
            return 1
        return 0

"""
29. Divide Two Integers

Given two integers dividend and divisor, divide two integers without using multiplication, division and mod
operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero.

Example 1:
Input: dividend = 10, divisor = 3
Output: 3

Example 2:
Input: dividend = 7, divisor = -3
Output: -2

Note:
Both dividend and divisor will be 32-bit signed integers.
The divisor will never be 0.
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer
range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 231 − 1 when the
division result overflows.
"""

class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """   
        res = 0
        
        is_pos = True
        if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
            is_pos = False
        
        dividend, divisor = abs(dividend), abs(divisor)
        curr, count = divisor, 1
        
        while dividend >= divisor:
            while dividend < curr:
                curr >>= 1
                count >>= 1
            
            dividend -= curr
            res += count
            curr <<= 1
            count <<= 1
        
        if not is_pos:
            res = -res
        
        return min(2**31 - 1, max(-2**31, res))