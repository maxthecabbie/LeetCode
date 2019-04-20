"""
973. K Closest Points to Origin

We have a list of points on the plane.  Find the K closest points to the origin (0, 0).

(Here, the distance between two points on a plane is the Euclidean distance.)

You may return the answer in any order.  The answer is guaranteed to be unique (except for the order that it
is in.)

Example 1:
Input: points = [[1,3],[-2,2]], K = 1
Output: [[-2,2]]

Explanation: 
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest K = 1 points from the origin, so the answer is just [[-2,2]].

Example 2:
Input: points = [[3,3],[5,-1],[-2,4]], K = 2
Output: [[3,3],[-2,4]]
(The answer [[-2,4],[3,3]] would also be accepted.)
 
Note:
1 <= K <= points.length <= 10000
-10000 < points[i][0] < 10000
-10000 < points[i][1] < 10000
"""

class Solution(object):
    def kClosest(self, points, K):
        """
        :type points: List[List[int]]
        :type K: int
        :rtype: List[List[int]]
        """
        heap = []
        
        for p in points:
            d = (p[0] ** 2 + p[1] ** 2) ** 0.5
            heapq.heappush(heap, (-d, p))
            
            if len(heap) > K:
                heapq.heappop(heap)
        
        return [entry[1] for entry in heap]

"""
378. Kth Smallest Element in a Sorted Matrix

Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest
element in the matrix.

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
            heapq.heappush(heap, (matrix[0][j], 0, j))
        
        for _ in range(k-1):
            cell = heapq.heappop(heap)
            x, y = cell[1], cell[2]
            
            if x == len(matrix) - 1:
                continue
            heapq.heappush(heap, (matrix[x+1][y], x+1, y))
        
        return heapq.heappop(heap)[0]

"""
215. Kth Largest Element in an Array

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted 
order, not the kth distinct element.

Example 1:
Input: [3,2,1,5,6,4] and k = 2
Output: 5

Example 2:
Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4

Note: 
You may assume k is always valid, 1 ≤ k ≤ array's length.
"""

class Solution:
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        heap = []
        for n in nums:
            heapq.heappush(heap, n)
        
        for _ in range(len(nums) - k):
            heapq.heappop(heap)
        
        return heapq.heappop(heap)

# Alternative solution in O(n) average time using quickselect
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        return self.quick_select(nums, 0, len(nums) - 1, k)
    
    def quick_select(self, nums, lo, hi, k):
        if lo <= hi:
            p = self.partition(nums, lo, hi)
            
            if k-1 == p:
                return nums[p]
            elif k-1 < p:
                return self.quick_select(nums, lo, p-1, k)
            else:
                return self.quick_select(nums, p+1, hi, k)

    def partition(self, nums, lo, hi):
        swap = lo
        
        for i in range(lo, hi):
            if nums[i] > nums[hi]:
                nums[i], nums[swap] = nums[swap], nums[i]
                swap += 1
        
        nums[swap], nums[hi] = nums[hi], nums[swap]
        return swap

"""
692. Top K Frequent Words

Given a non-empty list of words, return the k most frequent elements.

Your answer should be sorted by frequency from highest to lowest. If two words have the same frequency, then
the word with the lower alphabetical order comes first.

Example 1:
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]

Explanation: "i" and "love" are the two most frequent words.
Note that "i" comes before "love" due to a lower alphabetical order.

Example 2:
Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]

Explanation: "the", "is", "sunny" and "day" are the four most frequent words,
with the number of occurrence being 4, 3, 2 and 1 respectively.

Note:
You may assume k is always valid, 1 ≤ k ≤ number of unique elements.
Input words contain only lowercase letters.

Follow up:
Try to solve it in O(n log k) time and O(n) extra space.
"""

class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """
        res = []
        
        counts = collections.Counter(words)
        heap = [(-counts[word], word) for word in counts]
        heapq.heapify(heap)
        
        for _ in range(k):
            res.append(heapq.heappop(heap)[1])
        
        return res

"""
253. Meeting Rooms II

Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), 
find the minimum number of conference rooms required.

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

