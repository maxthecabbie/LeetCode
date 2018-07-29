"""
535. Encode and Decode TinyURL

Note: This is a companion problem to the System Design problem: Design TinyURL.

TinyURL is a URL shortening service where you enter a URL such as https://leetcode.com/problems/
design-tinyurl and it returns a short URL such as http://tinyurl.com/4e9iAk.

Design the encode and decode methods for the TinyURL service. There is no restriction on how your encode/
decode algorithm should work. You just need to ensure that a URL can be encoded to a tiny URL and the tiny 
URL can be decoded to the original URL.
"""

class Codec:
    def __init__(self):
    	self.url_mappings = {}

    def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        end = []
        for i in range(6): 
            end.append(random.choice(chars))
        tinyurl = "http://www.tinyurl.com/" + "".join(end)
        self.url_mappings[tinyurl] = longUrl
        return tinyurl

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        if shortUrl not in self.url_mappings:
            return None
        return self.url_mappings[shortUrl]
        
# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))

"""
531. Lonely Pixel I

Given a picture consisting of black and white pixels, find the number of black lonely pixels.

The picture is represented by a 2D char array consisting of 'B' and 'W', which means black and white pixels 
respectively.

A black lonely pixel is character 'B' that located at a specific position where the same row and same column 
don't have any other black pixels.

Example:
Input: 
[['W', 'W', 'B'],
 ['W', 'B', 'W'],
 ['B', 'W', 'W']]

Output: 3
Explanation: All the three 'B's are black lonely pixels.

Note:
The range of width and height of the input 2D array is [1,500].

Solution Explanation
- Have 2 dictionaries that keep track of the number of "B" pixels for each row and column
- Iterate through the matrix and if the cell has a "B", record the row and column in the dictionaries
- Iterate through the matrix again and for each cell check that the row and column have only one "B" pixel. 
If it does, increase the count by 1
by 1
- Return the count
"""

class Solution:
    def findLonelyPixel(self, picture):
        """
        :type picture: List[List[str]]
        :rtype: int
        """
        rows = {}
        cols = {}
        count = 0
        
        for i in range(len(picture)):
            for j in range(len(picture[0])):
                if picture[i][j] == "B":
                    rows[i] = rows.get(i, 0) + 1
                    cols[j] = cols.get(j, 0) + 1
        
        for i in range(len(picture)):
            for j in range(len(picture[0])):
                if picture[i][j] == "B" and rows[i] == 1 and cols[j] == 1:
                    count += 1
        
        return count

"""
451. Sort Characters By Frequency
Given a string, sort it in decreasing order based on the frequency of characters.

Example 1:
Input:
"tree"

Output:
"eert"

Explanation:
'e' appears twice while 'r' and 't' both appear once.
So 'e' must appear before both 'r' and 't'. Therefore "eetr" is also a valid answer.

Example 2:
Input:
"cccaaa"

Output:
"cccaaa"

Explanation:
Both 'c' and 'a' appear three times, so "aaaccc" is also a valid answer.
Note that "cacaca" is incorrect, as the same characters must be together.

Example 3:
Input:
"Aabb"

Output:
"bbAa"

Explanation:
"bbaA" is also a valid answer, but "Aabb" is incorrect.
Note that 'A' and 'a' are treated as two different characters.
"""

class Solution:
    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        
        freq = {}
        buckets = [[] for _ in range(len(s) + 1)]
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        
        for key in freq:
            buckets[freq[key]].append(key)
        
        result = []
        for i in range(len(buckets) - 1, -1, -1):
            if len(buckets[i]) > 0:
                for c in buckets[i]:
                    result.append(i*c)
        return "".join(result)

"""
667. Beautiful Arrangement II

Given two integers n and k, you need to construct a list which contains n different positive integers 
ranging from 1 to n and obeys the following requirement: 
Suppose this list is [a1, a2, a3, ... , an], then the list [|a1 - a2|, |a2 - a3|, |a3 - a4|, ... , |an-1 - 
an|] has exactly k distinct integers.

If there are multiple answers, print any of them.

Example 1:
Input: n = 3, k = 1
Output: [1, 2, 3]
Explanation: The [1, 2, 3] has three different positive integers ranging from 1 to 3, and the [1, 1] has 
exactly 1 distinct integer: 1.

Example 2:
Input: n = 3, k = 2
Output: [1, 3, 2]
Explanation: The [1, 3, 2] has three different positive integers ranging from 1 to 3, and the [2, 1] has 
exactly 2 distinct integers: 1 and 2.

Note:
The n and k are in the range 1 <= k < n <= 104.

Solution Explanation
- The max k we can have is n-1 since we have the numbers 1 to n, so there are only 1 to n-1 possible 
differences we can make
- Consider if k is equal to n-1, the only possible arrangement is if 1 and n are next to each other, then 
n-1 comes after followed by 2 and so on
- We can leverage this pattern by realizing for the first n-k numbers, we can just put in 1 to n-k-1. These
will have difference 1 between them. Now for n-k to n, we can apply the pattern of n-k next to n, then n-k+1 
comes after followed by n-1 and so on
- This works because the number n-k next to n will have a difference of k, and n-k+1 next to n will have a 
difference of k-1, and n-k+1 next to n-1 will have a difference of k-2 and so on
"""

class Solution:
    def constructArray(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[int]
        """
        result = [i for i in range(1, n-k)]
        start = n - k
        end = n
        
        for i in range(k+1):
            if i % 2 == 0:
                result.append(start)
                start += 1
            else:
                result.append(end)
                end -= 1
        return result

"""
238. Product of Array Except Self

Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the 
product of all the elements of nums except nums[i].

Example:
Input:  [1,2,3,4]
Output: [24,12,8,6]

Note: Please solve it without division and in O(n).

Follow up:
Could you solve it with constant space complexity? (The output array does not count as extra space for the 
purpose of space complexity analysis.)
"""

class Solution:
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if len(nums) <= 0: return [0]
        
        l = 1
        r = 1
        result = [1 for _ in range(len(nums))]
        
        for i in range(len(nums)):
            result[i] *= l
            result[len(nums) - i - 1] *= r
            l *= nums[i]
            r *= nums[len(nums) - i - 1]
        
        return result

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
769. Max Chunks To Make Sorted

Given an array arr that is a permutation of [0, 1, ..., arr.length - 1], we split the array into some number 
of "chunks" (partitions), and individually sort each chunk.  After concatenating them, the result equals the 
sorted array.

What is the most number of chunks we could have made?

Example 1:
Input: arr = [4,3,2,1,0]
Output: 1
Explanation:
Splitting into two or more chunks will not return the required result.
For example, splitting into [4, 3], [2, 1, 0] will result in [3, 4, 0, 1, 2], which isn't sorted.

Example 2:
Input: arr = [1,0,2,3,4]
Output: 4
Explanation:
We can split into two chunks, such as [1, 0], [2, 3, 4].
However, splitting into [1, 0], [2], [3], [4] is the highest number of chunks possible.

Note:
arr will have length in range [1, 10].
arr[i] will be a permutation of [0, 1, ..., arr.length - 1].

Solution Explanation
- In this question, we need to keep track of the max number we have encountered, which initially is 0. The 
chunks variable keeps track of the chunks we can sort so far 
- At every index in our array, we first update max_num and then check if max_num is less than or equal to 
the current index
	- If it is, that means we can make a sorted chunk since the rest of the elements below it are less than 
    or equal to the max_num, and no number in a higher index is greater than max_num. Increase chunks by 1
    - Else, that means max_num is greater than the current index, and we cannot make a sorted chunk since if 
    we make a sorted chunk, then the max_num will not be in the correct index and will be out of place
- Return chunks
"""

class Solution:
    def maxChunksToSorted(self, arr):
        """
        :type arr: List[int]
        :rtype: int
        """
        max_num = 0
        chunks = 0
        
        for i in range(len(arr)):
            max_num = max(max_num, arr[i])
            
            if max_num <= i:
                chunks += 1
        
        return chunks

"""
454. 4Sum II

Given four lists A, B, C, D of integer values, compute how many tuples (i, j, k, l) there are such that A[i] 
+ B[j] + C[k] + D[l] is zero.

To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. All integers are in 
the range of -228 to 228 - 1 and the result is guaranteed to be at most 231 - 1.

Example:
Input:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

Output:
2

Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0

Solution Explanation
- For each number in A, loop through each number in B and create an entry in the ab_sum dictionary where the 
key is the sum of the two numbers and the value is how many pairs have that sum
those two numbers. If an entry already exists, add 1 to it, else set it to 1
- For each number in C, loop through each number in D and check if 0 minus the sum of those two numbers is
in the dictionary. If so, add the value to the count
- Return the count
"""

class Solution:
    def fourSumCount(self, A, B, C, D):
        """
        :type A: List[int]
        :type B: List[int]
        :type C: List[int]
        :type D: List[int]
        :rtype: int
        """
        ab_sum = {}
        count = 0
        
        for i in range(len(A)):
            for j in range(len(B)):
                ab = A[i] + B[j]
                ab_sum[ab] = ab_sum.get(ab, 0) + 1
        
        for i in range(len(C)):
            for j in range(len(D)):
                cd = C[i] + D[j]
                check = 0 - cd
                if check in ab_sum:
                    count += ab_sum[check]
        return count

"""
78. Subsets

Given a set of distinct integers, nums, return all possible subsets (the power set).

Note: The solution set must not contain duplicate subsets.

Example:
Input: nums = [1,2,3]
Output:
[
  [3],
  [1],
  [2],
  [1,2,3],
  [1,3],
  [2,3],
  [1,2],
  []
]
"""

class Solution:
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        result = [[]]
        
        for i in range(len(nums)):
            new_result = []
            for sub in result:
                cpy = sub[:]
                cpy.append(nums[i])
                new_result.append(cpy)
            result += new_result
        
        return result

"""
487. Max Consecutive Ones II

Given a binary array, find the maximum number of consecutive 1s in this array if you can flip at most one 0.

Example 1:
Input: [1,0,1,1,0]
Output: 4
Explanation: Flip the first zero will get the the maximum number of consecutive 1s.
    After flipping, the maximum number of consecutive 1s is 4.

Note:
The input array will only contain 0 and 1.
The length of input array is a positive integer and will not exceed 10,000

Follow up:
What if the input numbers come in one by one as an infinite stream? In other words, you can't store all 
numbers coming from the stream as it's too large to hold in memory. Could you solve it efficiently?

Solution Explanation
- Have 3 variables, max_count, count, and prev
- Iterate through array:
	- If value is 1, increase count by 1
	- If value is 0, update max_count and:
		- If previous was also a 0, set prev to 0
		- If previous was not a 0, set prev to count
		- Set count back to 0 in both cases
- Have to be careful to handle the case where entire array is all 1's, in which case we will have an answer 
that is 1 more than expected
"""

class Solution:
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_count = 0
        count = 0
        prev = 0
        
        for i in range(len(nums)):
            if nums[i] == 1:
                count += 1
            else:
                max_count = max(max_count, count + prev)
                if i > 0 and nums[i - 1] == 0:
                    prev = 0
                else:
                    prev = count
                count = 0
        
        return min(len(nums), max(max_count, count + prev) + 1)

"""
249. Group Shifted Strings

Given a string, we can "shift" each of its letter to its successive letter, for example: "abc" -> "bcd". We 
can keep "shifting" which forms the sequence:

"abc" -> "bcd" -> ... -> "xyz"
Given a list of strings which contains only lowercase alphabets, group all strings that belong to the same 
shifting sequence.

Example:

Input: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"],
Output: 
[
  ["abc","bcd","xyz"],
  ["az","ba"],
  ["acef"],
  ["a","z"]
]

Solution Explanation
- For every string we create a shift sequence, which is the difference in ord values of subsequent characters
- We have to be careful if we have something like "ba", since technically the sequence is the difference 
between a -> z and then z -> b
- Group strings by their shift sequence in a dictionary and output the result
"""

class Solution(object):
    def groupStrings(self, strings):
        """
        :type strings: List[str]
        :rtype: List[List[str]]
        """
        seqs = {}
        for s in strings:
            seq = self.get_seq(s)
            seqs.setdefault(seq, []).append(s)
        return [seqs[key] for key in seqs]
    
    def get_seq(self, s):
        if len(s) <= 1:
            return 0
        
        seq = []
        for i in range(1, len(s)):
            diff = ord(s[i]) - ord(s[i-1])
            if diff < 0:
                diff_to_z = ord(s[i]) - ord("a") + 1
                diff = diff_to_z + (ord("z") - ord(s[i-1]))
            seq.append(str(diff))
        return ":".join(seq)

"""
287. Find the Duplicate Number

Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that 
at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate 
one.

Example 1:
Input: [1,3,4,2,2]
Output: 2

Example 2:
Input: [3,1,3,4,2]
Output: 3

Note:
You must not modify the array (assume the array is read only).
You must use only constant, O(1) extra space.
Your runtime complexity should be less than O(n2).
There is only one duplicate number in the array, but it could be repeated more than once.
"""

class Solution:
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        fast = nums[0]
        slow = nums[0]

        while True:
        	fast = nums[nums[fast]]
        	slow = nums[slow]
        	if fast == slow:
        		break

        p1 = nums[0]
        p2 = slow

        while p1 != p2:
            p1 = nums[p1]
            p2 = nums[p2]
        
        return p1

"""
360. Sort Transformed Array

Given a sorted array of integers nums and integer values a, b and c. Apply a quadratic function of the form f
(x) = ax2 + bx + c to each element x in the array.

The returned array must be in sorted order.

Expected time complexity: O(n)

Example:
nums = [-4, -2, 2, 4], a = 1, b = 3, c = 5,

Result: [3, 9, 15, 33]

nums = [-4, -2, 2, 4], a = -1, b = 3, c = 5

Result: [-23, -5, 1, 7]

Solution Explanation
- For quadratic function what really matters is the sign of a:
    - If a is positive, then largest numbers will be largest positve and negative numbers, which are the 
    numbers at the ends of the array
    - If a is negative, the largest numbers will be in the middle since large positive and negative numbers 
    will produce large negative numbers
- Transform array using quadratic formula
- Keep two pointers, one at start, and one at end of nums array
- Initialize a result array the same size as nums array
- Keep an i variable, that keeps track of where we are inserting in the result array
    - If a is positive, then i starts at end of result array
    - If a is negative, then i starts at start of result array
- While pt1 is less than or equal to pt2, compare pointers and insert them into corresponding position in 
result.
- Return result array
"""

class Solution(object):
    def sortTransformedArray(self, nums, a, b, c):
        """
        :type nums: List[int]
        :type a: int
        :type b: int
        :type c: int
        :rtype: List[int]
        """
        nums = [a*(n*n) + b*n + c for n in nums]
        result = [0] * len(nums)
        pt1 = 0
        pt2 = len(nums) - 1
        i = 0 if a < 0 else len(nums) - 1
        
        while pt1 <= pt2:
            if a < 0:
                if nums[pt1] < nums[pt2]:
                    result[i] = nums[pt1]
                    pt1 += 1
                else:
                    result[i] = nums[pt2]
                    pt2 -= 1
                i += 1
            else:
                if nums[pt1] > nums[pt2]:
                    result[i] = nums[pt1]
                    pt1 += 1
                else:
                    result[i] = nums[pt2]
                    pt2 -= 1
                i -= 1
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

Solution Explanation
- We sort our dictionary by length and lexicographic order
- Then for each word in the dictionary, we see if that word is a subsequence of the string s, and if it is 
we return that word
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
325. Maximum Size Subarray Sum Equals k

Given an array nums and a target value k, find the maximum length of a subarray that sums to k. If there 
isn't one, return 0 instead.

Note:
The sum of the entire nums array is guaranteed to fit within the 32-bit signed integer range.

Example 1:
Input: nums = [1, -1, 5, -2, 3], k = 3
Output: 4 
Explanation: The subarray [1, -1, 5, -2] sums to 3 and is the longest.

Example 2:
Input: nums = [-2, -1, 2, 1], k = 1
Output: 2 
Explanation: The subarray [-1, 2] sums to 1 and is the longest.
Follow Up:
Can you do it in O(n) time?

Solution Explanation
- Iterate through the numbers in our nums array
- We, keep track of the cumulative sum, and for each number n we iterate over, we calculate k-n which is 
what we need some previous consecutive numbers to add up to
- If we can find a difference between the current cumulative sum and a previous cumulative sum such that it 
equals k-n, then we can consider that previous index for the max size subarray
    - Update the max size if the value of the index for that previous cumulative sum is greater than the 
    current max size
- Only update the cumulative sum if there is no such cumulative sum in the dictionary, because we want the 
lowest index of such a cumulative sum to give us a max size subarray
- Return max_size
"""

class Solution(object):
    def maxSubArrayLen(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        sums = {}
        max_size = 0
        curr_sum = 0
        
        for i in range(len(nums)):
            if nums[i] == k:
                max_size = max(max_size, 1)
                
            check = curr_sum - (k - nums[i])
            if check in sums:
                index = sums[check]
                max_size = max(max_size, i - index + 1)
            
            if curr_sum not in sums:
                sums[curr_sum] = i
            curr_sum += nums[i]

        return max_size

"""
48. Rotate Image

You are given an n x n 2D matrix representing an image.

Rotate the image by 90 degrees (clockwise).

Note:

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example 1:
Given input matrix = 
[
  [1,2,3],
  [4,5,6],
  [7,8,9]
],

rotate the input matrix in-place such that it becomes:
[
  [7,4,1],
  [8,5,2],
  [9,6,3]
]

Example 2:
Given input matrix =
[
  [ 5, 1, 9,11],
  [ 2, 4, 8,10],
  [13, 3, 6, 7],
  [15,14,12,16]
], 

rotate the input matrix in-place such that it becomes:
[
  [15,13, 2, 5],
  [14, 3, 4, 1],
  [12, 6, 8, 9],
  [16, 7,10,11]
]
"""

class Solution:
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        layers = len(matrix)//2
        layer_iterations = len(matrix) - 1
        end = len(matrix) - 1

        for i in range(layers):
            for j in range(layer_iterations):
                temp = matrix[i][i+j]
                matrix[i][i+j] = matrix[end-j][i]
                matrix[end-j][i] = matrix[end][end-j]
                matrix[end][end-j] = matrix[i+j][end]
                matrix[i+j][end] = temp
            end -= 1
            layer_iterations -= 2

"""
846. Hand of Straights

Alice has a hand of cards, given as an array of integers.

Now she wants to rearrange the cards into groups so that each group is size W, and consists of W consecutive cards.

Return true if and only if she can.

Example 1:
Input: hand = [1,2,3,6,2,3,4,7,8], W = 3
Output: true
Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8].

Example 2:
Input: hand = [1,2,3,4,5], W = 4
Output: false
Explanation: Alice's hand can't be rearranged into groups of 4.
 
Note:
1 <= hand.length <= 10000
0 <= hand[i] <= 10^9
1 <= W <= hand.length
"""

class Solution:
    def isNStraightHand(self, hand, W):
        freq = {}
        
        for n in hand:
            freq[n] = freq.get(n, 0) + 1
        
        hand.sort()
        
        for i in range(len(hand)):
            if hand[i] > 0:
                for j in range(W)[::-1]:
                    if hand[i + j] <= 0:
                        return False
                    hand[i + j] -= hand[i]
                    
        return True

"""
259. 3Sum Smaller

Given an array of n integers nums and a target, find the number of index triplets i, j, k with 0 <= i < j < k < n that satisfy the condition nums[i] + nums[j] + nums[k] < target.

Example:
Input: nums = [-2,0,1,3], and target = 2
Output: 2 
Explanation: Because there are two triplets which sums are less than 2:
             [-2,0,1]
             [-2,0,3]

Follow up: Could you solve it in O(n2) runtime?
"""

class Solution:
    def threeSumSmaller(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        nums.sort()
        count = 0
        
        for i in range(len(nums) - 2):
            left = i + 1
            right = len(nums) - 1
            
            while left < right:
                if nums[i] + nums[left] + nums[right] >= target:
                    right -= 1
                else:
                    count += right - left
                    left += 1
        
        return count

"""
681. Next Closest Time

Given a time represented in the format "HH:MM", form the next closest time by reusing the current digits. There is no limit on how many times a digit can be reused.

You may assume the given input string is always valid. For example, "01:34", "12:09" are all valid. "1:34", "12:9" are all invalid.

Example 1:
Input: "19:34"
Output: "19:39"
Explanation: The next closest time choosing from digits 1, 9, 3, 4, is 19:39, which occurs 5 minutes later.  It is not 19:33, because this occurs 23 hours and 59 minutes later.

Example 2:
Input: "23:59"
Output: "22:22"
Explanation: The next closest time choosing from digits 2, 3, 5, 9, is 22:22. It may be assumed that the returned time is next day

Solution Explanation
- Generate next closest time by starting at the right most number
- Sort the numbers we have from the time, and get the low and high numbers
- Starting at the right, see if there is a larger number in nums, then we can set that digit to the larger number and return the time
    - For m2, any larger number will work
    - For m1, the larger number has to be less than 6
    - For h2, the larger number can be any number if the h1 is less than 2, else less than 4
    - For h1, the larger number must be less than 3
- If you do set a digit to a larger number, must set all digits to the right of it to lowest to get the next lowest time
- If no next closer time can be generated, set all digits to lowest number and return it
"""

class Solution(object):
    def nextClosestTime(self, time):
        """
        :type time: str
        :rtype: str
        """
        nums = sorted([int(c) for c in time if c != ":"])
        lo, hi = nums[0], nums[-1]
        m1, m2 = int(time[3]), int(time[4])
        h1, h2 = int(time[0]), int(time[1])
        
        if m2 != hi:
            for n in nums:
                if n > m2: 
                    m2 = n
                    return str(h1) + str(h2) + ":" + str(m1) + str(m2)
        
        if m1 != hi:
            for n in nums:
                if n > m1 and n < 6:
                    m1 = n
                    m2 = lo
                    return str(h1) + str(h2) + ":" + str(m1) + str(m2)
        
        if h2 != hi:
            for n in nums:
                if n > h2 and not (h1 == 2 and n > 3):
                    h2 = n
                    m1, m2 = lo, lo
                    return str(h1) + str(h2) + ":" + str(m1) + str(m2)
        
        if h1 != hi:
            for n in nums:
                if n > h1 and n < 3 and not (h1 == 2):
                    h1 = n
                    h2, m1, m2 = lo, lo, lo
                    return str(h1) + str(h2) + ":" + str(m1) + str(m2)
        
        h1, h2, m1, m2 = lo, lo, lo, lo
        return str(h1) + str(h2) + ":" + str(m1) + str(m2)

"""
380. Insert Delete GetRandom O(1)

Design a data structure that supports all following operations in average O(1) time.

insert(val): Inserts an item val to the set if not already present.
remove(val): Removes an item val from the set if present.
getRandom: Returns a random element from current set of elements. Each element must have the same 
probability of being returned.

Example:
// Init an empty set.
RandomizedSet randomSet = new RandomizedSet();

// Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomSet.insert(1);

// Returns false as 2 does not exist in the set.
randomSet.remove(2);

// Inserts 2 to the set, returns true. Set now contains [1,2].
randomSet.insert(2);

// getRandom should return either 1 or 2 randomly.
randomSet.getRandom();

// Removes 1 from the set, returns true. Set now contains [2].
randomSet.remove(1);

// 2 was already in the set, so return false.
randomSet.insert(2);

// Since 2 is the only number in the set, getRandom always return 2.
randomSet.getRandom();

Solution Explanation
- Keep a dictionary that maps vals to indices in the vals array
- For insert, simply append to vals array and put index into indices
- For remove, look up index of val in vals array, swap last element into the index, and then pop last 
element from vals and delete entry the val entry in indices
- For getRandom, simply return random element from vals array
"""

class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.indices = {}
        self.vals = []

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val in self.indices:
            return False
        
        index = len(self.vals)
        self.vals.append(val)
        self.indices[val] = index
        return True
        
    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.indices:
            return False
        index = self.indices[val]
        last = len(self.vals) - 1
        last_val = self.vals[last]

        self.vals[index] = last_val
        self.vals.pop()
        self.indices[last_val] = index
        del self.indices[val]
        return True

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return random.choice(self.vals)
        
# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()

"""
49. Group Anagrams

Given an array of strings, group anagrams together.

Example:
Input: ["eat", "tea", "tan", "ate", "nat", "bat"],
Output:
[
  ["ate","eat","tea"],
  ["nat","tan"],
  ["bat"]
]

Note:
All inputs will be in lowercase.
The order of your output does not matter.
"""

class Solution:
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        anagrams = {}
        
        for s in strs:
            key = [0 for _ in range(26)]
            for ch in s:
                i = ord(ch) - ord("a")
                key[i] += 1
            key = ":".join([str(n) for n in key])
            anagrams.setdefault(key, []).append(s)
        
        result = []
        for key in anagrams:
            result.append(anagrams[key])
        
        return result

"""
560. Subarray Sum Equals K

Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose 
sum equals to k.

Example 1:
Input:nums = [1,1,1], k = 2
Output: 2

Note:
The length of the array is in range [1, 20,000].
The range of numbers in the array is [-1000, 1000] and the range of the integer k is [-1e7, 1e7].

Solution Explanation
- We keep track of the cumulative sum as we iterate through nums, storing the cumulative sum in a dictionary 
as the key, and the number of indexes that have this cumulative sum
- At each number n in nums, we calculate k-n, which is what we need some previous consecutive numbers to add 
up to before we reach the number n
- To find such consecutive numbers, we take the current cumulative sum and subtract the number we need our  
consecutive numbers to add up to, which is k-n
- Thus we ask: from our current cumulative sum, I need to find a cumulative sum from a previous index such 
that the difference between the current cumulative sum and that previous cumulative sum is k-n. If we can 
find a previous cumulative sum where adding k-n to it results in our current cumulative sum, then the 
difference between the two plus our current number n will give us the target
- So now we check to see if we have such a previous cumulative sum, and if so, add the value from the 
dictionary to our count
- Also add 1 to our count if the current number equals our target
- At the end of each loop, put the current cumulative sum in the hash table and update the cumulative sum
- Return the count
"""

class Solution:
    def subarraySum(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        cu_sums = {}
        cu_sum = 0
        count = 0
        
        for i in range(len(nums)):
            if nums[i] == k:
                count += 1
            
            need = k - nums[i]
            cu_sum_diff = cu_sum - need
            count += cu_sums.get(cu_sum_diff, 0)
            
            cu_sums[cu_sum] = cu_sums.get(cu_sum, 0) + 1
            cu_sum += nums[i]

        return count

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
A rather straight forward solution is a two-pass algorithm using counting sort.
First, iterate the array counting number of 0's, 1's, and 2's, then overwrite array with total number of 
0's, then 1's and followed by 2's.
Could you come up with a one-pass algorithm using only constant space?
"""

class Solution:
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        red = 0
        white = 0
        blue = 0
        
        for n in nums:
            if n == 0:
                red += 1
            elif n == 1:
                white += 1
            else:
                blue += 1
        
        i = 0
        for _ in range(red):
            nums[i] = 0
            i += 1
        for _ in range(white):
            nums[i] = 1
            i += 1
        for _ in range(blue):
            nums[i] = 2
            i += 1

"""
731. My Calendar II

Implement a MyCalendarTwo class to store your events. A new event can be added if adding the event will not 
cause a triple booking.

Your class will have one method, book(int start, int end). Formally, this represents a booking on the half 
open interval [start, end), the range of real numbers x such that start <= x < end.

A triple booking happens when three events have some non-empty intersection (ie., there is some time that is
 common to all 3 events.)

For each call to the method MyCalendar.book, return true if the event can be added to the calendar 
successfully without causing a triple booking. Otherwise, return false and do not add the event to the 
calendar.

Your class will be called like this: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)

Example 1:
MyCalendar();
MyCalendar.book(10, 20); // returns true
MyCalendar.book(50, 60); // returns true
MyCalendar.book(10, 40); // returns true
MyCalendar.book(5, 15); // returns false
MyCalendar.book(5, 10); // returns true
MyCalendar.book(25, 55); // returns true
Explanation: 
The first two events can be booked.  The third event can be double booked.
The fourth event (5, 15) can't be booked, because it would result in a triple booking.
The fifth event (5, 10) can be booked, as it does not use time 10 which is already double booked.
The sixth event (25, 55) can be booked, as the time in [25, 40) will be double booked with the third event;
the time [40, 50) will be single booked, and the time [50, 55) will be double booked with the second event.

Note:
The number of calls to MyCalendar.book per test case will be at most 1000.
In calls to MyCalendar.book(start, end), start and end are integers in the range [0, 10^9].


Solution Explanation
- Store two arrays, one for events booked and another for overlapping events
- For every call to book, check if the start and end time overlap with an entry in overlaps, if it does, 
return False
- If no triple bookings, check for a double booking, and add the max start time and min end time as a new
 entry to overlaps
- Finally add the start and end time to the calendar
- Return True
- NOTE how we check for overlaps
"""
class MyCalendarTwo(object):

    def __init__(self):
        self.overlaps = []
        self.calendar = []

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        for olap in self.overlaps:
            if not (end <= olap[0] or start >= olap[1]):
                return False
        for entry in self.calendar:
            if not (end <= entry[0] or start >= entry[1]):
                self.overlaps.append([max(start, entry[0]), min(end, entry[1])])
        self.calendar.append([start, end])
        return True

# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)

"""
731. My Calendar II

Implement a MyCalendarTwo class to store your events. A new event can be added if adding the event will not 
cause a triple booking.

Your class will have one method, book(int start, int end). Formally, this represents a booking on the half 
open interval [start, end), the range of real numbers x such that start <= x < end.

A triple booking happens when three events have some non-empty intersection (ie., there is some time that is 
common to all 3 events.)

For each call to the method MyCalendar.book, return true if the event can be added to the calendar 
successfully without causing a triple booking. Otherwise, return false and do not add the event to the 
calendar.

Your class will be called like this: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)

Example 1:
MyCalendar();
MyCalendar.book(10, 20); // returns true
MyCalendar.book(50, 60); // returns true
MyCalendar.book(10, 40); // returns true
MyCalendar.book(5, 15); // returns false
MyCalendar.book(5, 10); // returns true
MyCalendar.book(25, 55); // returns true
Explanation: 
The first two events can be booked.  The third event can be double booked.
The fourth event (5, 15) can't be booked, because it would result in a triple booking.
The fifth event (5, 10) can be booked, as it does not use time 10 which is already double booked.
The sixth event (25, 55) can be booked, as the time in [25, 40) will be double booked with the third event;
the time [40, 50) will be single booked, and the time [50, 55) will be double booked with the second event.

Note:
The number of calls to MyCalendar.book per test case will be at most 1000.
In calls to MyCalendar.book(start, end), start and end are integers in the range [0, 10^9].
"""

class MyCalendarTwo(object):

    def __init__(self):
        self.overlaps = []
        self.calendar = []

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        for olap in self.overlaps:
            if (start < olap[1] and olap[0] < end):
                return False
            
        for entry in self.calendar:
            if not (end <= entry[0] or start >= entry[1]):
                self.overlaps.append([max(start, entry[0]), min(end, entry[1])])
                
        self.calendar.append([start, end])
        return True

# Your MyCalendarTwo object will be instantiated and called as such:
# obj = MyCalendarTwo()
# param_1 = obj.book(start,end)

"""
36. Valid Sudoku

Determine if a 9x9 Sudoku board is valid. Only the filled cells need to be validated according to the 
following rules:

Each row must contain the digits 1-9 without repetition.
Each column must contain the digits 1-9 without repetition.
Each of the 9 3x3 sub-boxes of the grid must contain the digits 1-9 without repetition.

The Sudoku board could be partially filled, where empty cells are filled with the character '.'.

Example 1:
Input:
[
  ["5","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
Output: true

Example 2:
Input:
[
  ["8","3",".",".","7",".",".",".","."],
  ["6",".",".","1","9","5",".",".","."],
  [".","9","8",".",".",".",".","6","."],
  ["8",".",".",".","6",".",".",".","3"],
  ["4",".",".","8",".","3",".",".","1"],
  ["7",".",".",".","2",".",".",".","6"],
  [".","6",".",".",".",".","2","8","."],
  [".",".",".","4","1","9",".",".","5"],
  [".",".",".",".","8",".",".","7","9"]
]
Output: false
Explanation: Same as Example 1, except with the 5 in the top left corner being 
    modified to 8. Since there are two 8's in the top left 3x3 sub-box, it is invalid.

Note:
A Sudoku board (partially filled) could be valid but is not necessarily solvable.
Only the filled cells need to be validated according to the mentioned rules.
The given board contain only digits 1-9 and the character '.'.
The given board size is always 9x9.

https://leetcode.com/problems/valid-sudoku/description/
"""

class Solution:
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        return self.valid_rows(board) and self.valid_cols(board)\
        and self.valid_boxes(board)
    
    def valid_rows(self, board):
        for row in board:
            if not self.check(row):
                return False
        return True

    
    def valid_cols(self, board):
        for j in range(len(board[0])):
            col = [board[i][j] for i in range(len(board))]
            if not self.check(col):
                return False
        return True
    
    def valid_boxes(self, board):
        for i in (0, 3, 6):
            for j in (0, 3, 6):
                box = [board[r][c] for r in range(i, i+3) for c in range(j, j+3)]
                if not self.check(box):
                    return False
        return True
    
    def check(self, nums):
        seen = set()
        for n in nums:
            if n != "." and n in seen:
                return False
            seen.add(n)
        return True

"""
289. Game of Life

According to the Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton 
devised by the British mathematician John Horton Conway in 1970."

Given a board with m by n cells, each cell has an initial state live (1) or dead (0). Each cell interacts 
with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the 
above Wikipedia article):

Any live cell with fewer than two live neighbors dies, as if caused by under-population.
Any live cell with two or three live neighbors lives on to the next generation.
Any live cell with more than three live neighbors dies, as if by over-population..
Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
Write a function to compute the next state (after one update) of the board given its current state. The next 
state is created by applying the above rules simultaneously to every cell in the current state, where births 
and deaths occur simultaneously.

Example:
Input: 
[
  [0,1,0],
  [0,0,1],
  [1,1,1],
  [0,0,0]
]
Output: 
[
  [0,0,0],
  [1,0,1],
  [0,1,1],
  [0,1,0]
]

Follow up:
Could you solve it in-place? Remember that the board needs to be updated at the same time: You cannot update some cells first and then use their updated values to update other cells.
In this question, we represent the board using a 2D array. In principle, the board is infinite, which would cause problems when the active area encroaches the border of the array. How would you address these problems?
"""

class Solution:
    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if len(board) <= 0: return
    
        for i in range(len(board)):
            for j in range(len(board[0])):
                live = self.get_live(i, j, board)
                if (live < 2 or live > 3) and board[i][j] == 1:
                    board[i][j] = 2
                if live == 3 and board[i][j] == 0: 
                    board[i][j] = 3
                        
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 2:
                    board[i][j] = 0
                elif board[i][j] == 3:
                    board[i][j] = 1
    
    def get_live(self, i, j, board):
        live = 0
        rows = len(board)
        cols = len(board[0])
        
        for r in range(i-1, i+2):
            for c in range(j-1, j+2):
                if (r >= 0 and r < rows and \
                    c >= 0 and c < cols and not \
                   (r == i and c == j)):
                    if board[r][c] == 1 or board[r][c] == 2:
                        live += 1
        return live
            
"""
616. Add Bold Tag in String

Given a string s and a list of strings dict, you need to add a closed pair of bold tag <b> and </b> to wrap 
the substrings in s that exist in dict. If two such substrings overlap, you need to wrap them together by 
only one pair of closed bold tag. Also, if two substrings wrapped by bold tags are consecutive, you need to 
combine them.

Example 1:
Input: 
s = "abcxyz123"
dict = ["abc","123"]
Output:
"<b>abc</b>xyz<b>123</b>"

Example 2:
Input: 
s = "aaabbcc"
dict = ["aaa","aab","bc"]
Output:
"<b>aaabbc</b>c"

Note:
The given dict won't contain duplicates, and its length won't exceed 100.
All the strings in input have length in range [1, 1000].

Solution Explanation
- Initialize array b_map to be the size of s where all values are 0's. Keeps track of which letters in s are 
enclosed in a bold tag
- For each word in dict, we keep searching for that word in s, and starting from the index of that word in s 
for the length of the word, we set those positions to 1 in b_map
- Then we iterate through s:
    - If that position is 1 in b_map, we place an opening <b> tag. We then keep looping and appending the 
    characters to our result until we encounter an index that is 0 in b_map or we reach the end of the 
    string. Then we place a closing tag
    - If that position is 0 in b_map we simply append that character to our result
- Return the result array joined as a string
"""

class Solution:
    def addBoldTag(self, s, dict):
        """
        :type s: str
        :type dict: List[str]
        :rtype: str
        """
        b_map = [0 for _ in range(len(s))]
        for w in dict:
            i = s.find(w)
            while i != -1:
                for j in range(i, i + len(w)):
                    b_map[j] = 1
                i = s.find(w, i + 1)
        i = 0
        result = []
        
        while i < len(s):
            if b_map[i] == 1:
                result.append("<b>")
                while i < len(s) and b_map[i] == 1:
                    result.append(s[i])
                    i += 1
                result.append("</b>")
            else:
                result.append(s[i])
                i += 1
                
        return "".join(result)

"""
17. Letter Combinations of a Phone Number

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number 
could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map 
to any letters.

Example:

Input: "23"
Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].

Note:
Although the above answer is in lexicographical order, your answer could be in any order you want.

https://leetcode.com/problems/letter-combinations-of-a-phone-number/description/
"""

class Solution:
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if len(digits) <= 0:
            return []
        
        num_letts = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz"
        }
        
        result = [ch for ch in num_letts[digits[0]]]
        
        for d in digits[1:]:
            new_result = []
            letters = num_letts[d]
            for l in letters:
                for res in result:
                    new_result.append(res+l)
            result = new_result
        
        return result

"""
11. Container With Most Water

Given n non-negative integers a1, a2, ..., an , where each represents a point at coordinate (i, ai). n 
vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, 
which together with x-axis forms a container, such that the container contains the most water.

Note: You may not slant the container and n is at least 2.

Example:
Input: [1,8,6,2,5,4,8,3,7]
Output: 49

https://leetcode.com/problems/container-with-most-water/description/
"""

class Solution:
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        l = 0
        r = len(height) - 1
        max_area = 0
        
        while l < r:
            max_area = max(max_area, (r-l) * min(height[r], height[l]))
            
            if height[l] < height[r]:
                l += 1
            else:
                r -= 1
        
        return max_area

"""
379. Design Phone Directory

Design a Phone Directory which supports the following operations:

get: Provide a number which is not assigned to anyone.
check: Check if a number is available or not.
release: Recycle or release a number.

Example:
// Init a phone directory containing a total of 3 numbers: 0, 1, and 2.
PhoneDirectory directory = new PhoneDirectory(3);

// It can return any available phone number. Here we assume it returns 0.
directory.get();

// Assume it returns 1.
directory.get();

// The number 2 is available, so return true.
directory.check(2);

// It returns 2, the only number that is left.
directory.get();

// The number 2 is no longer available, so return false.
directory.check(2);

// Release number 2 back to the pool.
directory.release(2);

// Number 2 is available again, return true.
directory.check(2);
"""

class PhoneDirectory(object):

    def __init__(self, maxNumbers):
        """
        Initialize your data structure here
        @param maxNumbers - The maximum numbers that can be stored in the phone directory.
        :type maxNumbers: int
        """
        self.nums = {i for i in range(maxNumbers)}

    def get(self):
        """
        Provide a number which is not assigned to anyone.
        @return - Return an available number. Return -1 if none is available.
        :rtype: int
        """
        if len(self.nums) <= 0: return -1
        return self.nums.pop()
        

    def check(self, number):
        """
        Check if a number is available or not.
        :type number: int
        :rtype: bool
        """
        return number in self.nums

    def release(self, number):
        """
        Recycle or release a number.
        :type number: int
        :rtype: void
        """
        self.nums.add(number)

# Your PhoneDirectory object will be instantiated and called as such:
# obj = PhoneDirectory(maxNumbers)
# param_1 = obj.get()
# param_2 = obj.check(number)
# obj.release(number)

"""
73. Set Matrix Zeroes

Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in-place.

Example 1:
Input: 
[
  [1,1,1],
  [1,0,1],
  [1,1,1]
]
Output: 
[
  [1,0,1],
  [0,0,0],
  [1,0,1]
]

Example 2:
Input: 
[
  [0,1,2,0],
  [3,4,5,2],
  [1,3,1,5]
]
Output: 
[
  [0,0,0,0],
  [0,4,5,0],
  [0,3,1,0]
]

Follow up:
A straight forward solution using O(mn) space is probably a bad idea.
A simple improvement uses O(m + n) space, but still not the best solution.
Could you devise a constant space solution?
"""

class Solution:
    def setZeroes(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        first_col = False

        for i in range(len(matrix)):
            if matrix[i][0] == 0:
                first_col = True
            for j in range(1, len(matrix[0])):
                if matrix[i][j] == 0:
                    matrix[i][0], matrix[0][j] = 0, 0

        for i in range(len(matrix) - 1, - 1, - 1):
            for j in range(1, len(matrix[0])):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
            if first_col:
                matrix[i][0] = 0

"""
792. Number of Matching Subsequences

Given string S and a dictionary of words words, find the number of words[i] that is a subsequence of S.

Example:
Input: 
S = "abcde"
words = ["a", "bb", "acd", "ace"]
Output: 3
Explanation: There are three words in words that are a subsequence of S: "a", "acd", "ace".

Note:
All words in words and S will only consists of lowercase letters.
The length of S will be in the range of [1, 50000].
The length of words will be in the range of [1, 5000].
The length of words[i] will be in the range of [1, 50].

Solution Explanation
- For every word in words, create iterators and store them in the words array
- Create an array called heads which has an array for each letter from a-z, and stores all iterators that 
are on that character
- For every iterator, store it in the corresponding array in heads
- For each character in S:
    - Get the corresponding array in heads that has all iterators that are on that character
        - For each iterator in the array, use next to advance iterator, or return None if it is the end of 
        the string
            - Put the iterator in the corresponding array in heads if there is a next character
            - Else increase count n by 1
- Return count n
"""

class Solution(object):
    def numMatchingSubseq(self, S, words):
        """
        :type S: str
        :type words: List[str]
        :rtype: int
        """
        heads = [[] for _ in range(26)]
        words = [iter(w) for w in words]
        
        for w in words:
            start = w.next()
            heads[ord(start) - ord("a")].append(w)
        
        n = 0
        for c in S:
            head_list = heads[ord(c) - ord("a")]
            heads[ord(c) - ord("a")] = []
            for w in head_list:
                next_char =next(w, None)
                if next_char is not None:
                    heads[ord(next_char) - ord("a")].append(w)
                else:
                    n += 1
        return n

"""
299. Bulls and Cows

You are playing the following Bulls and Cows game with your friend: You write down a number and ask your 
friend to guess what the number is. Each time your friend makes a guess, you provide a hint that indicates 
how many digits in said guess match your secret number exactly in both digit and position (called "bulls") 
and how many digits match the secret number but locate in the wrong position (called "cows"). Your friend 
will use successive guesses and hints to eventually derive the secret number.

Write a function to return a hint according to the secret number and friend's guess, use A to indicate the 
bulls and B to indicate the cows. 

Please note that both secret number and friend's guess may contain duplicate digits.

Example 1:
Input: secret = "1807", guess = "7810"

Output: "1A3B"

Explanation: 1 bull and 3 cows. The bull is 8, the cows are 0, 1 and 7.

Example 2:
Input: secret = "1123", guess = "0111"

Output: "1A1B"

Explanation: The 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow.
Note: You may assume that the secret number and your friend's guess only contain digits, and their lengths 
are always equal.

Solution Explanation
- Iterate through characters in secret and keep track of the frequency of characters in a dictionary
- Iterate through the guess characters:
    - If the character at index i is the same as the character in the secret at index i then increment bull 
    count by 1
        - If the value of that character in the freq dictionary is 0, then you must decrement the number of 
        colws
        - Else just decrement the value of that character in the freq dictionary
    - If the character is in the freq dictionary and the value is greater than 0, then increment the cow 
    count and decrement the value in the freq dictionary by 1
- Return the bull and cow count
"""

class Solution:
    def getHint(self, secret, guess):
        """
        :type secret: str
        :type guess: str
        :rtype: str
        """
        freq = {}
        b, c = 0, 0
        
        for ch in secret:
            freq[ch] = freq.get(ch, 0) + 1
            
        for i in range(len(guess)):
            ch = guess[i]
            if ch == secret[i]:
                b += 1
                if freq[ch] == 0:
                    c -= 1
                else:
                    freq[ch] -= 1
            elif freq.get(ch, 0) > 0:
                c += 1
                freq[ch] -= 1
        
        return str(b) + "A" + str(c) + "B"

"""
274. H-Index

Given an array of citations (each citation is a non-negative integer) of a researcher, write a function to 
compute the researcher's h-index.

According to the definition of h-index on Wikipedia: "A scientist has index h if h of his/her N papers have 
at least h citations each, and the other N − h papers have no more than h citations each."

Example:
Input: citations = [3,0,6,1,5]
Output: 3 
Explanation: [3,0,6,1,5] means the researcher has 5 papers in total and each of them had 
             received 3, 0, 6, 1, 5 citations respectively. 
             Since the researcher has 3 papers with at least 3 citations each and the remaining 
             two with no more than 3 citations each, her h-index is 3.

Note: If there are several possible values for h, the maximum one is taken as the h-index.
"""

class Solution:
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        if len(citations) <= 0:
            return 0
        
        h_index = 0
        citations.sort(reverse = True)

        for i in range(len(citations)):
            n = citations[i]
            if n <= i+1:
                return n
            else:
                return min(i+1, n)
        
        return h_index

# Alternative solution in O(n) time
class Solution:
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        counts = [0] * (len(citations) + 1)
        
        for i in range(len(citations)):
            if citations[i] >= len(counts):
                counts[-1] += 1
            else:
                counts[citations[i]] += 1
                
        total = 0
        for i in range(len(counts) - 1, -1, -1):
            total += counts[i]
            if total >= i:
                return i
        return 0

"""
228. Summary Ranges

Given a sorted integer array without duplicates, return the summary of its ranges.

Example 1:
Input:  [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
Explanation: 0,1,2 form a continuous range; 4,5 form a continuous range.

Example 2:
Input:  [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
Explanation: 2,3,4 form a continuous range; 8,9 form a continuous range.
"""

class Solution:
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        result = []
        last = 0

        for i in range(len(nums)):
            if i+1 < len(nums) and nums[i+1] - nums[i] > 1:
                self.append_range(last, i, nums, result)
                last = i+1
        self.append_range(last, len(nums) - 1, nums, result)
        return result

    def append_range(self, last, i, nums, result):
        if len(nums) <= 0:
            return
        if last == i:
            result.append(str(nums[i]))
        else:
            result.append(str(nums[last]) + "->" + str(nums[i]))

"""
56. Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

Example 1:
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

Example 2:
Input: [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considerred overlapping.
"""

class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        intervals.sort(key = lambda i: i.start)
        result = []

        for i in intervals:
            if result and i.start <= result[-1].end:
                prev = result[-1]
                prev.end = max(prev.end, i.end)
            else:
                result.append(i)

        return result

"""
56. Merge Intervals

Given a collection of intervals, merge all overlapping intervals.

Example 1:
Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

Example 2:
Input: [[1,4],[4,5]]
Output: [[1,5]]
Explanation: Intervals [1,4] and [4,5] are considerred overlapping.
"""

class Solution:
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        intervals.sort(key = lambda i: i.start)
        result = []

        for i in intervals:
            if result and i.start <= result[-1].end:
                prev = result[-1]
                prev.end = max(prev.end, i.end)
            else:
                result.append(i)

        return result

"""
186. Reverse Words in a String II

Given an input string , reverse the string word by word. 

Example:
Input:  ["t","h","e"," ","s","k","y"," ","i","s"," ","b","l","u","e"]
Output: ["b","l","u","e"," ","i","s"," ","s","k","y"," ","t","h","e"]

Note: 
A word is defined as a sequence of non-space characters.
The input string does not contain leading or trailing spaces.
The words are always separated by a single space.
Follow up: Could you do it in-place without allocating extra space?

Solution Explanation
- Create a copy of the str array
- Loop through the copy backwards. The end variable keeps track of the current word being reversed
- When a " " is encountered or we are at the beginning of the str array, we set our start variable to be the 
start of the string to be reversed
- Then we copy the string into the str array using the ptr variable to keep track of where to insert the 
string
- If it isn't the last string, we append a " " and increase the str pointer by 1
- The end variable is set to i-1, since this will be the end of the next string
"""

class Solution:
    def reverseWords(self, str):
        """
        :type str: List[str]
        :rtype: void Do not return anything, modify str in-place instead.
        """
        cpy = str.copy()
        end = len(str) - 1
        ptr = 0
        
        for i in range(len(str) - 1, -1, -1):
            if cpy[i] == " " or i == 0:
                start = 0 if i == 0 else i + 1

                for j in range(start, end + 1):
                    str[ptr] = cpy[j]
                    ptr += 1
                if i != 0:
                    str[ptr] = " "
                    ptr += 1
                    
                end = i - 1


# Alternative solution using O(1) space
"""
- In this solution we simply reverse the entire array
- Then for each word in the array, we reverse that word
"""
class Solution:
    def reverseWords(self, str):
        """
        :type str: List[str]
        :rtype: void Do not return anything, modify str in-place instead.
        """
        str = self.reverse(str, 0, len(str) - 1)
        start = 0
        
        for i in range(len(str)):
            if str[i] == " " or i == len(str) - 1:
                end = i if i == len(str) - 1 else i - 1
                self.reverse(str, start, end)
                start = i + 1
    
    def reverse(self, str, i, j):        
        while i < j:
            temp = str[i]
            str[i] = str[j]
            str[j] = temp
            i += 1
            j -= 1
        return str

"""
161. One Edit Distance

Given two strings s and t, determine if they are both one edit distance apart.

Note: 
There are 3 possiblities to satisify one edit distance apart:

Insert a character into s to get t
Delete a character from s to get t
Replace a character of s to get t

Example 1:
Input: s = "ab", t = "acb"
Output: true
Explanation: We can insert 'c' into s to get t.

Example 2:
Input: s = "cab", t = "ad"
Output: false
Explanation: We cannot get t from s by only one step.

Example 3:
Input: s = "1203", t = "1213"
Output: true
Explanation: We can replace '0' with '1' to get t.

Solution Explanation
- Have two variables short, which is the shorter of s or t and long, which is the longer of s or t
    - NOTE, s and t might be the same size which we must account for as shown below
- Have two pointers for short and long which start at 0
- If we find a difference
    - If there was a previous difference, return False
    - Else:
        - If short and long are actually the same length then increase the pointers of both
        - Else we increase the pointer of long
- If no difference found increase the pointers of both strings
- Return True
"""

class Solution:
    def isOneEditDistance(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if abs(len(s) - len(t)) > 1:
            return False
        if s == t:
            return False
        
        short = s if len(s) <= len(t) else t
        long = s if len(s) > len(t) else t
        
        pt_s = 0
        pt_l = 0
        diff = False
        
        while pt_s < len(short):
            if short[pt_s] != long[pt_l]:
                if diff:
                    return False
                diff = True
                
                if len(short) == len(long):
                    pt_s += 1
                    pt_l += 1
                    continue
                pt_l += 1
                
            else:
                pt_s += 1
                pt_l += 1
        
        return True

"""
845. Longest Mountain in Array

Let's call any (contiguous) subarray B (of A) a mountain if the following properties hold:

B.length >= 3
There exists some 0 < i < B.length - 1 such that B[0] < B[1] < ... B[i-1] < B[i] > B[i+1] > ... > B[B.length 
- 1]
(Note that B could be any subarray of A, including the entire array A.)

Given an array A of integers, return the length of the longest mountain. 

Return 0 if there is no mountain.

Example 1:
Input: [2,1,4,7,3,2,5]
Output: 5
Explanation: The largest mountain is [1,4,7,3,2] which has length 5.

Example 2:
Input: [2,2,2]
Output: 0
Explanation: There is no mountain.

Note:
0 <= A.length <= 10000
0 <= A[i] <= 10000
Follow up:

Can you solve it using only one pass?
Can you solve it in O(1) space?

Solution Explanation
- Solution revolves around finding a valley, which is an element that is smaller than the surrounding two 
elements. We keep track of last_valley, which is the index of the last valley we have encountered. We update 
the longest mountain found when we are on the "downward" side of a valid mountain
- Loop through array:
    - If we encounter an element that is greater than 0 and is the same value as the previous element, we 
    set last_valley to None since this breaks our definition of a mountain
    - If the current index is greater than 0, the last element is larger than our current element, and 
    last_valey is not None, then we can update longest if the current index - last_valley + 1 is greater 
    than longest. NOTE we can only update longest when we are at an element that is smaller than the 
    previous element since if the elements only increase, it is not yet a mountain
    - If we are at index 0 or the previous element is greater than our current element AND if we are not at 
    the end of the array and the next element is greater than our current element THEN we can set this index 
    as last_valley
- Then we return longest if it is greater than or equal to 3, else we return 0
"""

class Solution:
    def longestMountain(self, arr):
        """
        :type A: List[int]
        :rtype: int
        """
        longest = 0
        last_valley = None
        
        for i in range(len(arr)):
            if i > 0 and arr[i] == arr[i-1]:
                last_valley = None
                
            elif i > 0 and arr[i] < arr[i-1] and last_valley is not None:
                longest = max(longest, i - last_valley + 1)
                    
            if i > 0 and arr[i] <= arr[i-1] or i == 0:
                if i + 1 < len(arr) and arr[i] < arr[i+1]:
                    last_valley = i
                    
        return longest if longest >= 3 else 0
            
"""
31. Next Permutation

Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

The replacement must be in-place and use only constant extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.

1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1
"""
class Solution:
    def nextPermutation(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        rev = 0

        for i in range(len(nums) - 2, -1, -1):
            if nums[i] < nums[i+1]:
                rev = i+1
                swap = len(nums) - 1
                while nums[swap] <= nums[i]:
                    swap -= 1
                nums[i], nums[swap] = nums[swap], nums[i]
                break

        end = len(nums) - 1
        while rev < end:
            nums[rev], nums[end] = nums[end], nums[rev]
            rev += 1
            end -= 1

"""
54. Spiral Matrix

Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.

Example 1:
Input:
[
 [ 1, 2, 3 ],
 [ 4, 5, 6 ],
 [ 7, 8, 9 ]
]
Output: [1,2,3,6,9,8,7,4,5]

Example 2:
Input:
[
  [1, 2, 3, 4],
  [5, 6, 7, 8],
  [9,10,11,12]
]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
"""

class Solution:
    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if len(matrix) <= 0:
            return []
        
        result = []
        
        layers = int(math.ceil(min(len(matrix), len(matrix[0]))/2))
        x_len = len(matrix[0])
        y_len = len(matrix) - 2
        x_end = len(matrix[0]) - 1
        y_end = len(matrix) - 1
        
        for i in range(layers):
            for j in range(x_len):
                result.append(matrix[i][i+j])
            
            for j in range(y_len):
                result.append(matrix[i+1+j][x_end])
            
            if i != y_end:
                for j in range(x_len):
                    result.append(matrix[y_end][x_end-j])
                    
            if i != x_end:
                for j in range(y_len):
                    result.append(matrix[y_end-1-j][i])
            
            x_len -= 2
            y_len -= 2
            x_end -= 1
            y_end -= 1
        
        return result

"""
271. Encode and Decode Strings

Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the 
network and is decoded back to the original list of strings.

Machine 1 (sender) has the function:

string encode(vector<string> strs) {
  // ... your code
  return encoded_string;
}

Machine 2 (receiver) has the function:
vector<string> decode(string s) {
  //... your code
  return strs;
}

So Machine 1 does:
string encoded_string = encode(strs);

and Machine 2 does:
vector<string> strs2 = decode(encoded_string);
strs2 in Machine 2 should be the same as strs in Machine 1.

Implement the encode and decode methods.

Note:
The string may contain any possible characters out of 256 valid ascii characters. Your algorithm should be 
generalized enough to work on any possible characters.
Do not use class member/global/static variables to store states. Your encode and decode algorithms should be 
stateless.
Do not rely on any library method such as eval or serialize methods. You should implement your own encode/
decode algorithm.

Solution Explanation
- To encode a string, for every string in the input strs, append to result <length>:<string>. The result 
string will consist of strings encoded in this format
- In the decode function, start with i = 0 which will be a digit. While the current character is a digit, 
keep increasing i and getting the value of the length of the string
- When this is done, i will be at the ":", so increase i by 1 to get the start of the string. The end of the 
string will be at start + len_s, so s[start:end] is the value of the encoded string which we can append to 
our result
- Set i to be the end, which is the start of the length of the next string
- Return result
"""

class Codec:

    def encode(self, strs):
        """Encodes a list of strings to a single string.
        
        :type strs: List[str]
        :rtype: str
        """
        result = []
        for s in strs:
            result.append(str(len(s)) + ":" + s)
        return "".join(result)

    def decode(self, s):
        """Decodes a single string to a list of strings.
        
        :type s: str
        :rtype: List[str]
        """
        result = []
        i = 0
        while i < len(s):
            len_s = 0
            while s[i].isdigit():
                len_s = 10 * len_s + int(s[i])
                i += 1
            start = i + 1
            end = start + len_s
            string = s[start:end]
            i = end
            result.append(string)
        return result

"""
5. Longest Palindromic Substring

Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s 
is 1000.

Example 1:
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.

Example 2:
Input: "cbbd"
Output: "bb"
"""

class Solution:
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        longest = s[:1]
        
        for i in range(len(s)):
            if i+1 < len(s) and s[i] == s[i+1]:
                pal = self.get_pal(i, i+1, s)
                longest = pal if len(pal) > len(longest) else longest
                
            if i+2 < len(s) and s[i] == s[i+2]:
                pal = self.get_pal(i, i+2, s)
                longest = pal if len(pal) > len(longest) else longest
        
        return longest
        
    def get_pal(self, i, j, s):
        while i >= 0 and j < len(s):
            if s[i] != s[j]:
                break
            i -= 1
            j += 1
        return s[i+1:j]

"""
3. Longest Substring Without Repeating Characters

Given a string, find the length of the longest substring without repeating characters.

Examples:
Given "abcabcbb", the answer is "abc", which the length is 3.

Given "bbbbb", the answer is "b", with the length of 1.

Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" 
is a subsequence and not a substring.
"""

class Solution:
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        seen = {}
        p1 = 0
        p2 = 0

        max_substring = 0
        count = 1
        
        while p2 < len(s):
            if s[p2] in seen:
                i = seen[s[p2]]
                while p1 <= i:
                    del seen[s[p1]]
                    p1 += 1
                    count -= 1
                    
            max_substring = max(max_substring, count)
            seen[s[p2]] = p2
            count += 1
            p2 += 1
        
        return max_substring
        
"""
163. Missing Ranges

Given a sorted integer array nums, where the range of elements are in the inclusive range [lower, upper], 
return its missing ranges.

Example:
Input: nums = [0, 1, 3, 50, 75], lower = 0 and upper = 99,
Output: ["2", "4->49", "51->74", "76->99"]

Solution Explanation
- last variable keeps track of the last number encountered, which starts at lower - 1
- Iterate through the array, if the current value minus last is greater than 1, we need to add a missing 
range
- The missing range will be between last + 1 and the current value - 1
    - Be careful, if last + 1 and current value - 1 are the same, we only need to add in one digit
    - If they are not the same value, we need to add the range
- Lastly, check that last is equal to upper after coming out from loop. If not then add the missing range
- Return result
"""

class Solution(object):
    def findMissingRanges(self, nums, lower, upper):
        """
        :type nums: List[int]
        :type lower: int
        :type upper: int
        :rtype: List[str]
        """
        result = []
        last = lower - 1

        for i in range(len(nums)):
            if nums[i] - last > 1:
                self.add_missing(last, nums[i], result)
            last = nums[i]
        
        if last != upper:
            self.add_missing(last, upper + 1, result)
        return result
    
    def add_missing(self, last, curr, result):
        start = last + 1
        end = curr - 1
        missing = str(start)
        if start != end:
            missing += "->" + str(end)
        result.append(missing)

"""
15. 3Sum

Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all 
unique triplets in the array which gives the sum of zero.

Note:
The solution set must not contain duplicate triplets.

Example:
Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
"""

class Solution:
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        num_ind = {}
        result = []
        
        for i in range(len(nums)):
            num_ind[nums[i]] = i
        
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            
            for j in range(i+1, len(nums) - 1):
                if nums[j] == nums[j-1] and j-1 != i:
                    continue

                check = -(nums[i] + nums[j])
                if check in num_ind and num_ind[check] > j:
                    result.append([nums[i], nums[j], nums[num_ind[check]]])
            
        return result

"""
165. Compare Version Numbers

Compare two version numbers version1 and version2.
If version1 > version2 return 1; if version1 < version2 return -1;otherwise return 0.

You may assume that the version strings are non-empty and contain only digits and the character. The 
character does not represent a decimal point and is used to separate number sequences. For instance, 2.5 is 
not "two and a half" or "half way to version three", it is the fifth second-level revision of the second 
first-level revision.

Example 1:
Input: version1 = "0.1", version2 = "1.1"
Output: -1

Example 2:
Input: version1 = "1.0.1", version2 = "1"
Output: 1

Example 3:
Input: version1 = "7.5.2.4", version2 = "7.5.3"
Output: -1
"""

class Solution:
    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        v1 = version1.split(".")
        v2 = version2.split(".")
        i = 0
        
        while i < len(v1) or i < len(v2):
            n1 = 0 if i >= len(v1) else int(v1[i])
            n2 = 0 if i >= len(v2) else int(v2[i])
            
            if n1 > n2:
                return 1
            elif n1 < n2:
                return -1
            i += 1
            
        return 0


"""
288. Unique Word Abbreviation

An abbreviation of a word follows the form <first letter><number><last letter>. Below are some examples of 
word abbreviations:

a) it                      --> it    (no abbreviation)

     1
     ↓
b) d|o|g                   --> d1g

              1    1  1
     1---5----0----5--8
     ↓   ↓    ↓    ↓  ↓    
c) i|nternationalizatio|n  --> i18n

              1
     1---5----0
     ↓   ↓    ↓
d) l|ocalizatio|n          --> l10n
Assume you have a dictionary and given a word, find whether its abbreviation is unique in the dictionary. A 
word's abbreviation is unique if no other word from the dictionary has the same abbreviation.

Example:
Given dictionary = [ "deer", "door", "cake", "card" ]

isUnique("dear") -> false
isUnique("cart") -> true
isUnique("cane") -> false
isUnique("make") -> true

Solution Explanation
- Create a dictionary of sets storing the abbreviation as a key where the set contains words with that 
abbreviation
- For isUnique:
    - If the abbreviation of the word is in the dictionary
        - If the set has length 1 and the word in the set is the same as the word given to the function, 
        return True
        - Else return False
    - Else return if the abbreviation of the word is not in the dictionary
"""

class ValidWordAbbr(object):

    def __init__(self, dictionary):
        """
        :type dictionary: List[str]
        """
        self.ab_dict = {}
        
        for word in dictionary:
            ab = self.get_ab(word)
            self.ab_dict.setdefault(ab, set()).add(word)

    def isUnique(self, word):
        """
        :type word: str
        :rtype: bool
        """
        ab = self.get_ab(word)
        
        if ab in self.ab_dict:
            if len(self.ab_dict[ab]) == 1 and word in self.ab_dict[ab]:
                return True
            return False
        
        return ab not in self.ab_dict
            
    
    def get_ab(self, word):
        if len(word) < 3:
            return word
        
        start = word[0]
        end = word[-1]
        
        return start + str(len(word) - 2) + end

# Your ValidWordAbbr object will be instantiated and called as such:
# obj = ValidWordAbbr(dictionary)
# param_1 = obj.isUnique(word)

"""
166. Fraction to Recurring Decimal

Given two integers representing the numerator and denominator of a fraction, return the fraction in string 
format.

If the fractional part is repeating, enclose the repeating part in parentheses.

Example 1:
Input: numerator = 1, denominator = 2
Output: "0.5"

Example 2:
Input: numerator = 2, denominator = 1
Output: "2"

Example 3:
Input: numerator = 2, denominator = 3
Output: "0.(6)"
"""

class Solution:
    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        """
        if numerator == 0: 
            return "0"
        
        result = []
        rem = {}
        if numerator * denominator < 0: 
            result.append("-")
        num = abs(numerator)
        den = abs(denominator)
        
        n = num % den
        res = num//den
        result.append(str(res))
        if res > 0 and n == 0: 
            return "".join(result)
        
        result.append(".")
        pos = len(result)
        
        while n:
            if n not in rem:
                rem[n] = pos
            else:
                repeat = "".join(result[rem[n]:])
                result = result[:rem[n]] + ["(" + repeat + ")"]
                break
            n *= 10
            res = n//den
            result.append(str(res))
            n %= den
            pos += 1
            
        return "".join(result)

"""
8. String to Integer (atoi)

Implement atoi which converts a string to an integer.

The function first discards as many whitespace characters as necessary until the first non-whitespace 
character is found. Then, starting from this character, takes an optional initial plus or minus sign 
followed by as many numerical digits as possible, and interprets them as a numerical value.

The string can contain additional characters after those that form the integral number, which are ignored 
and have no effect on the behavior of this function.

If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such 
sequence exists because either str is empty or it contains only whitespace characters, no conversion is 
performed.

If no valid conversion could be performed, a zero value is returned.

Note:
Only the space character ' ' is considered as whitespace character.
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer 
range: [−231,  231 − 1]. If the numerical value is out of the range of representable values, INT_MAX (231 − 
1) or INT_MIN (−231) is returned.

Example 1:
Input: "42"
Output: 42

Example 2:
Input: "   -42"
Output: -42
Explanation: The first non-whitespace character is '-', which is the minus sign.
             Then take as many numerical digits as possible, which gets 42.

Example 3:
Input: "4193 with words"
Output: 4193
Explanation: Conversion stops at digit '3' as the next character is not a numerical digit.

Example 4:
Input: "words and 987"
Output: 0
Explanation: The first non-whitespace character is 'w', which is not a numerical 
             digit or a +/- sign. Therefore no valid conversion could be performed.

Example 5:
Input: "-91283472332"
Output: -2147483648
Explanation: The number "-91283472332" is out of the range of a 32-bit signed integer.
             Thefore INT_MIN (−231) is returned.
"""

class Solution:
    def myAtoi(self, s):
        """
        :type s: s
        :rtype: int
        """
        s = s.strip()
        
        if len(s) <= 0:
            return 0
        
        sign = 1
        i = 0
        n = 0

        if s[0] == "-" or s[i] == "+":
            if s[0] == "-":
                sign = -1
            i += 1

        while i < len(s) and s[i].isdigit():
            n = 10 * n + int(s[i])
            i += 1

        return min(2**31 - 1, max(-2**31, sign * n))