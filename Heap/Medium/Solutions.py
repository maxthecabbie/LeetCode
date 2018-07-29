"""
378. Kth Smallest Element in a Sorted Matrix

Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

Example:
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

return 13.

Note: 
You may assume k is always valid, 1 ≤ k ≤ n2.
"""

class Solution:
    def kthSmallest(self, matrix, k):
        """
        :type matrix: List[List[int]]
        :type k: int
        :rtype: int
        """
        heap = []
        for j in range(len(matrix[0])):
            heapq.heappush(heap, [matrix[0][j], 0, j])
        
        for i in range(k-1):
            t = heapq.heappop(heap)
            x, y = t[1], t[2]
            
            if t[1] == len(matrix) - 1:
                continue
            heapq.heappush(heap, [matrix[x+1][y], x+1, y])
            
        return heapq.heappop(heap)[0]

"""
215. Kth Largest Element in an Array

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

Example 1:
Input: [3,2,1,5,6,4] and k = 2
Output: 5

Example 2:
Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4

Note: 
You may assume k is always valid, 1 ≤ k ≤ array's length.
"""

def findKthLargest4(self, nums, k):
    heap = []
    
    for num in nums:
        heapq.heappush(heap, num)
        
    for _ in xrange(len(nums) - k):
        heapq.heappop(heap)
        
    return heapq.heappop(heap)

# Alternative solution in O(n) average time using quickselect
class Solution:
    def findKthLargest(self, nums, k):
        return self.findKthSmallest(nums, len(nums) + 1 - k)

    def findKthSmallest(self, nums, k):
        if nums:
            pos = self.partition(nums, 0, len(nums) - 1)
            if k > pos + 1:
                return self.findKthSmallest(nums[pos + 1:], k - pos - 1)
            elif k < pos + 1:
                return self.findKthSmallest(nums[:pos], k)
            else:
                return nums[pos]

    def partition(self, nums, l, r):
        lo = l
        
        while l < r:
            if nums[l] < nums[r]:
                nums[l], nums[lo] = nums[lo], nums[l]
                lo += 1
            l += 1
            
        nums[lo], nums[r] = nums[r], nums[lo]
        return low

"""
253. Meeting Rooms II

Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.

Example 1:
Input: [[0, 30],[5, 10],[15, 20]]
Output: 2

Example 2:
Input: [[7,10],[2,4]]
Output: 1
"""

# Definition for an interval.
# class Interval:
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e
class Solution:
    def minMeetingRooms(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: int
        """
        heap = []
        intervals.sort(key = lambda x: x.start)
        
        for i in intervals:
            if heap and i.start >= heap[0]:
                heapq.heapreplace(heap, i.end)
            else:
                heapq.heappush(heap, i.end)
        
        return len(heap)            

"""
373. Find K Pairs with Smallest Sums

You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.

Define a pair (u,v) which consists of one element from the first array and one element from the second array.

Find the k pairs (u1,v1),(u2,v2) ...(uk,vk) with the smallest sums.

Example 1:
Given nums1 = [1,7,11], nums2 = [2,4,6],  k = 3

Return: [1,2],[1,4],[1,6]

The first 3 pairs are returned from the sequence:
[1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]

Example 2:
Given nums1 = [1,1,2], nums2 = [1,2,3],  k = 2

Return: [1,1],[1,1]

The first 2 pairs are returned from the sequence:
[1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]

Example 3:
Given nums1 = [1,2], nums2 = [3],  k = 3 

Return: [1,3],[2,3]

All possible pairs are returned from the sequence:
[1,3],[2,3]
"""

class Solution:
    def kSmallestPairs(self, nums1, nums2, k):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :type k: int
        :rtype: List[List[int]]
        """
        heap = []
        result = []
        n, m = len(nums1), len(nums2)
        
        if n <= 0 or m <= 0 or k <= 0:
            return result
        
        for i in range(n):
            heapq.heappush(heap, [nums1[i] + nums2[0], i, 0])
        
        for i in range(min(k, n*m)):
            t = heapq.heappop(heap)
            x, y = t[1], t[2]
            result.append([nums1[x], nums2[y]])
            
            if y == m - 1:
                continue
            heapq.heappush(heap, [nums1[x] + nums2[y+1], x, y+1])
        
        return result

