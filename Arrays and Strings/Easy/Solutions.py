"""
771. Jewels and Stones

You're given strings J representing the types of stones that are jewels, and S representing the stones you 
have.  Each character in S is a type of stone you have.  You want to know how many of the stones you have 
are also jewels.

The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case 
sensitive, so "a" is considered a different type of stone from "A".

Example 1:
Input: J = "aA", S = "aAAbbbb"
Output: 3

Example 2:
Input: J = "z", S = "ZZ"
Output: 0

Note:
S and J will consist of letters and have length at most 50.
The characters in J are distinct.
"""
class Solution:
    def numJewelsInStones(self, J, S):
        """
        :type J: str
        :type S: str
        :rtype: int
        """
        n = 0
        jset = set(J)
        for c in S:
            if c in jset: n += 1
        return n

"""
760. Find Anagram Mappings

Given two lists Aand B, and B is an anagram of A. B is an anagram of A means B is made by randomizing the 
order of the elements in A.

We want to find an index mapping P, from A to B. A mapping P[i] = j means the ith element in A appears in B 
at index j.

These lists A and B may contain duplicates. If there are multiple answers, output any of them.

For example, given
A = [12, 28, 46, 32, 50]
B = [50, 12, 32, 46, 28]

We should return
[1, 4, 3, 2, 0]
as P[0] = 1 because the 0th element of A appears at B[1], and P[1] = 4 because the 1st element of A appears 
at B[4], and so on.

Note:
A, B have equal lengths in range [1, 100].
A[i], B[i] are integers in range [0, 10^5].
"""
class Solution:
    def anagramMappings(self, A, B):
        """
        :type A: List[int]
        :type B: List[int]
        :rtype: List[int]
        """
        bmap = {v: i for i, v in enumerate(B)}
        return [bmap[v] for v in A]     

"""
709. To Lower Case

Implement function ToLowerCase() that has a string parameter str, and returns the same string in lowercase.

Example 1:

Input: "Hello"
Output: "hello"

Example 2:
Input: "here"
Output: "here"

Example 3:
Input: "LOVELY"
Output: "lovely"
"""

class Solution:
    def toLowerCase(self, s):
        """
        :type str: str
        :rtype: str
        """
        d = ord("A") - ord("a")
        res = [chr(ord(c) - d) if c.isalpha() and ord(c) < ord("a") else c for c in s]
        return "".join(res)


"""
804. Unique Morse Code Words
International Morse Code defines a standard encoding where each letter is mapped to a series of dots and 
dashes, as follows: "a" maps to ".-", "b" maps to "-...", "c" maps to "-.-.", and so on.

For convenience, the full table for the 26 letters of the English alphabet is given below:

[".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.
","...","-","..-","...-",".--","-..-","-.--","--.."]

Now, given a list of words, each word can be written as a concatenation of the Morse code of each letter. 
For example, "cab" can be written as "-.-.-....-", (which is the concatenation "-.-." + "-..." + ".-"). 
We'll call such a concatenation, the transformation of a word.

Return the number of different transformations among all words we have.

Example:
Input: words = ["gin", "zen", "gig", "msg"]
Output: 2
Explanation: 
The transformation of each word is:
"gin" -> "--...-."
"zen" -> "--...-."
"gig" -> "--...--."
"msg" -> "--...--."

There are 2 different transformations, "--...-." and "--...--.".
"""
class Solution:
    def uniqueMorseRepresentations(self, words):
        """
        :type words: List[str]
        :rtype: int
        """
        transformations = set()
        for w in words:
            morse = self.translate(w)
            if morse not in transformations: 
                transformations.add(morse)
        return len(transformations)
    
    def translate(self, word):
        alphabet = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---",
                    "-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-",
                    "...-",".--","-..-","-.--","--.."]
        return "".join([alphabet[ord(c) - ord("a")] for c in word])

"""
832. Flipping an Image

Given a binary matrix A, we want to flip the image horizontally, then invert it, and return the resulting 
image.

To flip an image horizontally means that each row of the image is reversed.  For example, flipping [1, 1, 0] 
horizontally results in [0, 1, 1].

To invert an image means that each 0 is replaced by 1, and each 1 is replaced by 0. For example, inverting [
0, 1, 1] results in [1, 0, 0].

Example 1:
Input: [[1,1,0],[1,0,1],[0,0,0]]
Output: [[1,0,0],[0,1,0],[1,1,1]]
Explanation: First reverse each row: [[0,1,1],[1,0,1],[0,0,0]].
Then, invert the image: [[1,0,0],[0,1,0],[1,1,1]]

Example 2:
Input: [[1,1,0,0],[1,0,0,1],[0,1,1,1],[1,0,1,0]]
Output: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]
Explanation: First reverse each row: [[0,0,1,1],[1,0,0,1],[1,1,1,0],[0,1,0,1]].
Then invert the image: [[1,1,0,0],[0,1,1,0],[0,0,0,1],[1,0,1,0]]

Notes:
1 <= A.length = A[0].length <= 20
0 <= A[i][j] <= 1
"""
class Solution:
    def flipAndInvertImage(self, A):
        """
        :type A: List[List[int]]
        :rtype: List[List[int]]
        """
        for row in A:
            for i in range((len(A[0]) + 1)//2):
                row[i], row[~i] = row[~i] ^ 1, row[i] ^ 1
        return A

"""
657. Judge Route Circle

Initially, there is a Robot at position (0, 0). Given a sequence of its moves, judge if this robot makes a 
circle, which means it moves back to the original place.

The move sequence is represented by a string. And each move is represent by a character. The valid robot 
moves are R (Right), L (Left), U (Up) and D (down). The output should be true or false representing whether 
the robot makes a circle.

Example 1:
Input: "UD"
Output: true

Example 2:
Input: "LL"
Output: false
"""
class Solution:
    def judgeCircle(self, moves):
        """
        :type moves: str
        :rtype: bool
        """
        p = 0
        for m in moves:
            if m == "L": 
                p += 1
            elif m == "R": 
                p -= 1
            elif m == "U": 
                p += 2
            elif m == "D": 
                p -= 2
        return p == 0

"""
561. Array Partition I

Given an array of 2n integers, your task is to group these integers into n pairs of integer, say (a1, b1), (
a2, b2), ..., (an, bn) which makes sum of min(ai, bi) for all i from 1 to n as large as possible.

Example 1:
Input: [1,4,3,2]

Output: 4
Explanation: n is 2, and the maximum sum of pairs is 4 = min(1, 2) + min(3, 4).

Note:
n is a positive integer, which is in the range of [1, 10000].
All the integers in the array will be in the range of [-10000, 10000].
"""

class Solution:
    def arrayPairSum(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        freq = [0] * 20001
        for n in nums:
            freq[n+10000] += 1
        
        result = 0
        prev = None
        
        for i in range(len(freq)):
            ind = len(freq) - i - 1
            while freq[ind] > 0:
                if prev is None:
                    prev = ind - 10000
                else:
                    curr = ind - 10000
                    result += min(prev, curr)
                    prev = None
                freq[ind] -= 1
                
        return result

"""
344. Reverse String

Write a function that takes a string as input and returns the string reversed.

Example:
Given s = "hello", return "olleh".
"""

class Solution:
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = [c for c in s]
        for i in range(len(s)//2):
            s[i], s[~i] = s[~i], s[i]
        return "".join(s)

"""
266. Palindrome Permutation

Given a string, determine if a permutation of the string could form a palindrome.

Example 1:
Input: "code"
Output: false

Example 2:
Input: "aab"
Output: true

Example 3:
Input: "carerac"
Output: true
"""

class Solution(object):
    def canPermutePalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        c_set = set()
        for c in s:
            if c in c_set:
                c_set.remove(c)
            else:
                c_set.add(c)
        return len(c_set) < 2

"""
766. Toeplitz Matrix

A matrix is Toeplitz if every diagonal from top-left to bottom-right has the same element.

Now given an M x N matrix, return True if and only if the matrix is Toeplitz.
 
Example 1:
Input:
matrix = [
  [1,2,3,4],
  [5,1,2,3],
  [9,5,1,2]
]
Output: True
Explanation:
In the above grid, the diagonals are:
"[9]", "[5, 5]", "[1, 1, 1]", "[2, 2, 2]", "[3, 3]", "[4]".
In each diagonal all elements are the same, so the answer is True.

Example 2:
Input:
matrix = [
  [1,2],
  [2,2]
]
Output: False
Explanation:
The diagonal "[1, 2]" has different elements.

Note:
matrix will be a 2D array of integers.
matrix will have a number of rows and columns in range [1, 20].
matrix[i][j] will be integers in range [0, 99].

Follow up:
What if the matrix is stored on disk, and the memory is limited such that you can only load at most one row 
of the matrix into the memory at once?
What if the matrix is so large that you can only load up a partial row into the memory at once?
"""

class Solution(object):
    def isToeplitzMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: bool
        """
        if len(matrix) <= 0: 
            return False
        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i-1 >= 0 and j-1 >= 0:
                    if matrix[i-1][j-1] != matrix[i][j]:
                        return False
        return True

"""
463. Island Perimeter
You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents 
water. Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded 
by water, and there is exactly one island (i.e., one or more connected land cells). The island doesn't have 
"lakes" (water inside that isn't connected to the water around the island). One cell is a square with side 
length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

Example:
[[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]

Answer: 16
Explanation: The perimeter is the 16 yellow stripes in the image below:

https://leetcode.com/problems/island-perimeter/description/
"""

class Solution:
    def islandPerimeter(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        p = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    if j == 0 or grid[i][j-1] == 0: 
                        p += 1
                    if i == 0 or grid[i-1][j] == 0: 
                        p += 1
                    if j == len(grid[0]) - 1 or grid[i][j+1] == 0:
                        p += 1
                    if i == len(grid) - 1 or grid[i+1][j] == 0:
                        p += 1
        return p

"""
412. Fizz Buzz

Write a program that outputs the string representation of numbers from 1 to n.

But for multiples of three it should output “Fizz” instead of the number and for the multiples of five 
output “Buzz”. For numbers which are multiples of both three and five output “FizzBuzz”.

Example:
n = 15,

Return:
[
    "1",
    "2",
    "Fizz",
    "4",
    "Buzz",
    "Fizz",
    "7",
    "8",
    "Fizz",
    "Buzz",
    "11",
    "Fizz",
    "13",
    "14",
    "FizzBuzz"
]
"""

class Solution:
    def fizzBuzz(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        result = []
        for i in range(1, n+1):
            if i % 3 == 0 and i % 5 == 0:
                result.append("FizzBuzz")
            elif i % 3 == 0:
                 result.append("Fizz")
            elif i % 5 == 0:
                 result.append("Buzz")
            else:
                 result.append(str(i))
        return result

"""
293. Flip Game

You are playing the following Flip Game with your friend: Given a string that contains only these two 
characters: + and -, you and your friend take turns to flip two consecutive "++" into "--". The game ends 
when a person can no longer make a move and therefore the other person will be the winner.

Write a function to compute all possible states of the string after one valid move.

Example:
Input: s = "++++"
Output: 
[
  "--++",
  "+--+",
  "++--"
]
Note: If there is no valid move, return an empty list [].
"""

class Solution:
    def generatePossibleNextMoves(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        result = []
        for i in range(1, len(s)):
            if s[i-1] == "+" and s[i] == "+":
                end = ""
                if i + 1 < len(s):
                    end = s[i+1:]
                result.append(s[:i-1] + "--" + end)
        return result

"""
136. Single Number

Given a non-empty array of integers, every element appears twice except for one. Find that single one.

Note:

Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

Example 1:
Input: [2,2,1]
Output: 1
Example 2:

Input: [4,1,2,1,2]
Output: 4
"""

class Solution:
    def singleNumber(self, arr):
        """
        :type nums: List[int]
        :rtype: int
        """
        num_set = set()
        sum_nums = 0
        for i in range(len(arr)):
            sum_nums += arr[i]
            num_set.add(arr[i])
        return sum(num_set) * 2 - sum_nums

"""
485. Max Consecutive Ones

Given a binary array, find the maximum number of consecutive 1s in this array.

Example 1:
Input: [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s.
    The maximum number of consecutive 1s is 3.

Note:
The input array will only contain 0 and 1.
The length of input array is a positive integer and will not exceed 10,000
"""

class Solution:
    def findMaxConsecutiveOnes(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        max_con = 0
        count = 0
        
        for i in range(len(nums)):
            if nums[i] == 1:
                count += 1
            else:
                max_con = max(max_con, count)
                count = 0
        
        return max(max_con, count)

"""
283. Move Zeroes

Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order 
of the non-zero elements.

Example:
Input: [0,1,0,3,12]
Output: [1,3,12,0,0]

Note:
You must do this in-place without making a copy of the array.
Minimize the total number of operations.
"""

class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        rep = 0
        
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[rep] = nums[i]
                rep += 1
        
        for i in range(rep, len(nums)):
            nums[i] = 0

# Alternative solution
class Solution:
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        rep = 0
        
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[i], nums[rep] = nums[rep], nums[i]
                rep += 1

"""
448. Find All Numbers Disappeared in an Array

Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others 
appear once.

Find all the elements of [1, n] inclusive that do not appear in this array.

Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as 
extra space.

Example:
Input:
[4,3,2,7,8,2,3,1]

Output:
[5,6]
"""

class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        # For each number i in nums,
        # we mark the number that i points as negative.
        # Then we filter the list, get all the indexes
        # who points to a positive number

        for i in range(len(nums)):
            ind = abs(nums[i]) - 1
            nums[ind] = - abs(nums[ind])
        return [i+1 for i in range(len(nums)) if nums[i] > 0]

"""
169. Majority Element

Given an array of size n, find the majority element. The majority element is the element that appears more 
than ⌊ n/2 ⌋ times.

You may assume that the array is non-empty and the majority element always exist in the array.

Example 1:
Input: [3,2,3]
Output: 3

Example 2:
Input: [2,2,1,1,1,2,2]
Output: 2
"""

class Solution:
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        freq = {}
        target = len(nums)//2
        
        for i in range(len(nums)):
            freq[nums[i]] = freq.get(nums[i], 0) + 1
        
        for k in freq:
            if freq[k] > target: 
                return k
"""
349. Intersection of Two Arrays

Given two arrays, write a function to compute their intersection.

Example:
Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2].

Note:
Each element in the result must be unique.
The result can be in any order.
"""

class Solution:
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        num_set = set()
        result = []
        
        for n in nums1:
            num_set.add(n)
        
        for n in nums2:
            if n in num_set:
                result.append(n)
                num_set.remove(n)
    
        return result

"""
217. Contains Duplicate

Given an array of integers, find if the array contains any duplicates.

Your function should return true if any value appears at least twice in the array, and it should return 
false if every element is distinct.

Example 1:

Input: [1,2,3,1]
Output: true
Example 2:

Input: [1,2,3,4]
Output: false
Example 3:

Input: [1,1,1,3,3,4,3,2,4,2]
Output: true
"""

class Solution:
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        num_set = set(nums)
        return len(num_set) != len(nums)

"""
387. First Unique Character in a String

Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, 
return -1.

Examples:

s = "leetcode"
return 0.

s = "loveleetcode",
return 2.
Note: You may assume the string contain only lowercase letters.
"""

class Solution:
    def firstUniqChar(self, s):
        """
        :type s: str
        :rtype: int
        """
        freq = [0] * 26
        
        for c in s:
            ind = ord(c) - ord("a")
            freq[ind] += 1
            
        for i in range(len(s)):
            ind = ord(s[i]) - ord("a")
            if freq[ind] == 1: 
                return i

        return -1

"""
268. Missing Number

Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from 
the array.

Example 1:
Input: [3,0,1]
Output: 2

Example 2:
Input: [9,6,4,2,3,5,7,0,1]
Output: 8

Note:
Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra 
space complexity?
"""

class Solution:
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        return sum(range(len(nums) + 1)) - sum(nums)

"""
167. Two Sum II - Input array is sorted

Given an array of integers that is already sorted in ascending order, find two numbers such that they add up 
to a specific target number.

The function twoSum should return indices of the two numbers such that they add up to the target, where 
index1 must be less than index2.

Note:
Your returned answers (both index1 and index2) are not zero-based.
You may assume that each input would have exactly one solution and you may not use the same element twice.
Example:

Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore index1 = 1, index2 = 2.
"""

class Solution:
    def twoSum(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        l, r = 0, len(numbers) - 1
        
        while l < r:
            tot = numbers[l] + numbers[r]
            if tot == target:
                return [l+1, r+1]
            if tot > target:
                r -= 1
            else:
                l += 1
        return [-1, -1]

"""
242. Valid Anagram

Given two strings s and t , write a function to determine if t is an anagram of s.

Example 1:
Input: s = "anagram", t = "nagaram"
Output: true

Example 2:
Input: s = "rat", t = "car"
Output: false
Note:
You may assume the string contains only lowercase alphabets.

Follow up:
What if the inputs contain unicode characters? How would you adapt your solution to such case?
"""

class Solution:
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t): 
            return False
        
        freq = {}
        
        for c in s:
            freq[c] = freq.get(c, 0) + 1
            
        for c in t:
            if freq.get(c, 0) > 0:
                freq[c] -= 1
            else:
                return False
            
        return True

"""
830. Positions of Large Groups

In a string S of lowercase letters, these letters form consecutive groups of the same character.

For example, a string like S = "abbxxxxzyy" has the groups "a", "bb", "xxxx", "z" and "yy".

Call a group large if it has 3 or more characters.  We would like the starting and ending positions of every
 large group.

The final answer should be in lexicographic order.

Example 1:
Input: "abbxxxxzzy"
Output: [[3,6]]
Explanation: "xxxx" is the single large group with starting  3 and ending positions 6.

Example 2:
Input: "abc"
Output: []
Explanation: We have "a","b" and "c" but no large group.

Example 3:
Input: "abcdddeeeeaabbbcd"
Output: [[3,5],[6,9],[12,14]]

Note:  1 <= S.length <= 1000
"""

class Solution:
    def largeGroupPositions(self, S):
        """
        :type S: str
        :rtype: List[List[int]]
        """
        result = []
        last = 0
        count = 1
        
        for i in range(1, len(S)):
            if S[i-1] == S[i]:
                count += 1
                
            if S[i-1] != S[i] or i == len(S) - 1 and count >= 3:
                if count >= 3:
                    result.append([last, last + count - 1])
                count = 1
                last = i
        
        return result

"""
844. Backspace String Compare

Given two strings S and T, return if they are equal when both are typed into empty text editors. # means a 
backspace character.

Example 1:
Input: S = "ab#c", T = "ad#c"
Output: true
Explanation: Both S and T become "ac".

Example 2:
Input: S = "ab##", T = "c#d#"
Output: true
Explanation: Both S and T become "".

Example 3:
Input: S = "a##c", T = "#a#c"
Output: true
Explanation: Both S and T become "c".

Example 4:
Input: S = "a#c", T = "b"
Output: false
Explanation: S becomes "c" while T becomes "b".

Note:
1 <= S.length <= 200
1 <= T.length <= 200
S and T only contain lowercase letters and '#' characters.

Follow up:
Can you solve it in O(N) time and O(1) space?
"""

class Solution:
    def backspaceCompare(self, S, T):
        """
        :type S: str
        :type T: str
        :rtype: bool
        """
        return self.type_str(S) == self.type_str(T)
    
    def type_str(self, s):
        res = []
        for c in s:
            if c.isalpha():
                res.append(c)
            elif res:
                res.pop()
        return "".join(res)

"""
350. Intersection of Two Arrays II

Given two arrays, write a function to compute their intersection.

Example:
Given nums1 = [1, 2, 2, 1], nums2 = [2, 2], return [2, 2].

Note:
Each element in the result should appear as many times as it shows in both arrays.
The result can be in any order.

Follow up:
What if the given array is already sorted? How would you optimize your algorithm?
What if nums1's size is small compared to nums2's size? Which algorithm is better?
What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all 
elements into the memory at once?
"""

class Solution:
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        freq = {}
        result = []
        
        for n in nums1:
            freq[n] = freq.get(n, 0) + 1
            
        for n in nums2:
            if n in freq and freq[n] > 0:
                result.append(n)
                freq[n] -= 1
                
        return result

#Follow up solution (nums1 and nums2 are both sorted)
class Solution:
    def intersect(self, nums1, nums2):
        result = []
        p2 = 0
        p1 = 0

        while p1 < len(nums1) and p2 < len(nums2):
            if nums1[p1] == nums2[p2]:
                result.append(nums1[p1])
                p1 += 1
                p2 += 1
            elif nums1[p1] < nums2[p2]: 
                p1 += 1
            else: 
                p2 += 1
                
        return result

"""
27. Remove Element

Given an array nums and a value val, remove all instances of that value in-place and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with 
O(1) extra memory.

The order of elements can be changed. It doesn't matter what you leave beyond the new length.

Example 1:
Given nums = [3,2,2,3], val = 3,

Your function should return length = 2, with the first two elements of nums being 2.

It doesn't matter what you leave beyond the returned length.

Example 2:
Given nums = [0,1,2,2,3,0,4,2], val = 2,

Your function should return length = 5, with the first five elements of nums containing 0, 1, 3, 0, and 4.

Note that the order of those five elements can be arbitrary.

It doesn't matter what values are set beyond the returned length.
Clarification:

Confused why the returned value is an integer but your answer is an array?

Note that the input array is passed in by reference, which means modification to the input array will be 
known to the caller as well.

Internally you can think of this:

// nums is passed in by reference. (i.e., without making a copy)
int len = removeElement(nums, val);

// any modification to nums in your function would be known by the caller.
// using the length returned by your function, it prints the first len elements.
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
"""

class Solution:
    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        swap = 0
        for i in range(len(nums)):
            if nums[i] != val:
                nums[swap], nums[i] = nums[i], nums[swap]
                swap += 1
        return swap

"""
747. Largest Number At Least Twice of Others

In a given integer array nums, there is always exactly one largest element.

Find whether the largest element in the array is at least twice as much as every other number in the array.

If it is, return the index of the largest element, otherwise return -1.

Example 1:
Input: nums = [3, 6, 1, 0]
Output: 1
Explanation: 6 is the largest integer, and for every other number in the array x,
6 is more than twice as big as x.  The index of value 6 is 1, so we return 1.
 

Example 2:
Input: nums = [1, 2, 3, 4]
Output: -1
Explanation: 4 isn't at least as big as twice the value of 3, so we return -1.
 

Note:
nums will have a length in the range [1, 50].
Every nums[i] will be an integer in the range [0, 99].
"""

class Solution:
    def dominantIndex(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) <= 0: 
            return -1
        
        hi = 0 
        sec_hi = -1
        
        for i in range(1, len(nums)):
            if nums[i] > nums[hi]:
                sec_hi = hi
                hi = i
            elif sec_hi == -1 or nums[i] > nums[sec_hi]:
                sec_hi = i
        
        valid_hi = sec_hi == -1 or nums[hi] >= nums[sec_hi] * 2
        return hi if valid_hi else -1

"""
66. Plus One

Given a non-empty array of digits representing a non-negative integer, plus one to the integer.

The digits are stored such that the most significant digit is at the head of the list, and each element in 
the array contain a single digit.

You may assume the integer does not contain any leading zero, except the number 0 itself.

Example 1:
Input: [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.

Example 2:
Input: [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.
"""

class Solution:
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        
        The key here is we only need to keep going
        until we encounter a number less than 9
        since if we add 1 to a number less than 9
        we do not need to carry 1 over to the next
        digit
        """
        for i in range(len(digits)):
            index = len(digits) - i - 1
            
            if digits[index] < 9:
                digits[index] += 1
                return digits
            
            digits[index] = 0
            
        digits.insert(0, 1)
        return digits

"""
1. Two Sum

Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:
Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
"""

class Solution:
    def twoSum(self, arr, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        num_ind = {}
        
        for i in range(len(arr)):
            check = target - arr[i]
            if check in num_ind: 
                return [num_ind[check], i]
            num_ind[arr[i]] = i
        
        return [-1, -1]

"""
643. Maximum Average Subarray I

Given an array consisting of n integers, find the contiguous subarray of given length k that has the maximum 
average value. And you need to output the maximum average value.

Example 1:
Input: [1,12,-5,-6,50,3], k = 4
Output: 12.75
Explanation: Maximum average is (12-5-6+50)/4 = 51/4 = 12.75

Note:
1 <= k <= n <= 30,000.
Elements of the given array will be in the range [-10,000, 10,000].
"""

class Solution:
    def findMaxAverage(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: float
        """
        curr = hi = sum(nums[:k])
        
        for i in range(len(nums)):
            if k+i < len(nums):
                curr += (nums[k+i]) - (nums[i])
                hi = max(hi, curr)
        return hi/min(k, len(nums))

"""
26. Remove Duplicates from Sorted Array
Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return 
the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with 
O(1) extra memory.

Example 1:
Given nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.

It doesn't matter what you leave beyond the returned length.

Example 2:
Given nums = [0,0,1,1,1,2,2,3,3,4],

Your function should return length = 5, with the first five elements of nums being modified to 0, 1, 2, 3, 
and 4 respectively.

It doesn't matter what values are set beyond the returned length.

Clarification:
Confused why the returned value is an integer but your answer is an array?

Note that the input array is passed in by reference, which means modification to the input array will be 
known to the caller as well.

Internally you can think of this:

// nums is passed in by reference. (i.e., without making a copy)
int len = removeDuplicates(nums);

// any modification to nums in your function would be known by the caller.
// using the length returned by your function, it prints the first len elements.
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
"""

class Solution:
    def removeDuplicates(self, arr):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(arr) <= 0: 
            return 0
        
        p = 1
        for i in range(1, len(arr)):
            if arr[i] != arr[i-1]:
                arr[p] = arr[i]
                p += 1
        return p

"""
303. Range Sum Query - Immutable

Given an integer array nums, find the sum of the elements between indices i and j (i ≤ j), inclusive.

Example:
Given nums = [-2, 0, 3, -5, 2, -1]

sumRange(0, 2) -> 1
sumRange(2, 5) -> -1
sumRange(0, 5) -> -3

Note:
You may assume that the array does not change.
There are many calls to sumRange function.
"""

class NumArray:

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.running_sums = [0]
        running_sum = 0
        for n in nums:
            running_sum += n
            self.running_sums.append(running_sum)
            
    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        return self.running_sums[j+1] - self.running_sums[i] 

# Your NumArray object will be instantiated and called as such:
# obj = NumArray(nums)
# param_1 = obj.sumRange(i,j)

"""
88. Merge Sorted Array

Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

Note:
The number of elements initialized in nums1 and nums2 are m and n respectively.
You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional 
elements from nums2.

Example:
Input:
nums1 = [1,2,3,0,0,0], m = 3
nums2 = [2,5,6],       n = 3

Output: [1,2,2,3,5,6]
"""

class Solution:
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        i = len(nums1) - 1
        p2 = n - 1
        p1 = m - 1
        
        while p2 >= 0 and p1 >= 0:
            if nums2[p2] >= nums1[p1]:
                nums1[i] = nums2[p2]
                p2 -= 1
            else:
                nums1[i] = nums1[p1]
                p1 -= 1
            i -= 1
         
        while p2 >= 0:
            nums1[i] = nums2[p2]
            p2 -= 1
            i -= 1

"""
14. Longest Common Prefix

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:
Input: ["flower","flow","flight"]
Output: "fl"

Example 2:
Input: ["dog","racecar","car"]
Output: ""
Explanation: There is no common prefix among the input strings.

Note:
All given inputs are in lowercase letters a-z.
"""

class Solution:
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if len(strs) <= 0: 
            return ""
        
        first = strs[0]
        
        for i in range(len(first)):
            for j in range(1, len(strs)):
                s = strs[j]
                if i == len(s) or first[i] != s[i]:
                    return first[:i]
            
        return first

"""
28. Implement strStr()

Implement strStr().

Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

Example 1:
Input: haystack = "hello", needle = "ll"
Output: 2

Example 2:
Input: haystack = "aaaaa", needle = "bba"
Output: -1

Clarification:
What should we return when needle is an empty string? This is a great question to ask during an interview.

For the purpose of this problem, we will return 0 when needle is an empty string. This is consistent to C's 
strstr() and Java's indexOf().
"""

class Solution:
    def strStr(self, hay, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        if len(needle) == 0: 
            return 0
        
        for i in range(len(hay)):
            if len(hay) - i < len(needle): 
                return -1
            for j in range(len(needle)):
                if hay[i+j] != needle[j]: 
                    break
                if j == len(needle) - 1: 
                    return i
        return -1

"""
157. Read N Characters Given Read4

The API: int read4(char *buf) reads 4 characters at a time from a file.

The return value is the actual number of characters read. For example, it returns 3 if there is only 3 
characters left in the file.

By using the read4 API, implement the function int read(char *buf, int n) that reads n characters from the 
file.

Example 1:
Input: buf = "abc", n = 4
Output: "abc"
Explanation: The actual number of characters read is 3, which is "abc".

Example 2:
Input: buf = "abcde", n = 5 
Output: "abcde"

Note:
The read function will only be called once for each test case.
"""

# The read4 API is already defined for you.
# @param buf, a list of characters
# @return an integer
# def read4(buf):

class Solution(object):
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Maximum number of characters to read (int)
        :rtype: The number of characters read (int)
        """
        total_read = 0
        
        while n > 0:
            buffer = ["" for _ in range(4)]
            
            chars = read4(buffer)
            
            if chars == 0:
                return total_read
            
            for i in range(min(chars, n)):
                buf[total_read] = buffer[i]
                total_read += 1
                n -= 1
            
        return total_read
        
"""
125. Valid Palindrome

Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

Note: For the purpose of this problem, we define empty string as valid palindrome.

Example 1:
Input: "A man, a plan, a canal: Panama"
Output: true

Example 2:
Input: "race a car"
Output: false
"""

class Solution:
    def isPalindrome(self, s):
        """
        :type s: str
        :rtype: bool
        """
        st = 0
        end = len(s) - 1
        
        while st < end:
            while not s[st].isalnum() and st < end:
                st += 1
            while not s[end].isalnum() and end > st:
                end -= 1
            
            if s[st].lower() != s[end].lower():
                return False
            st += 1
            end -= 1
            
        return True

""" 
189. Rotate Array
Given an array, rotate the array to the right by k steps, where k is non-negative.

Example 1:
Input: [1,2,3,4,5,6,7] and k = 3
Output: [5,6,7,1,2,3,4]
Explanation:
rotate 1 steps to the right: [7,1,2,3,4,5,6]
rotate 2 steps to the right: [6,7,1,2,3,4,5]
rotate 3 steps to the right: [5,6,7,1,2,3,4]

Example 2:
Input: [-1,-100,3,99] and k = 2
Output: [3,99,-1,-100]
Explanation: 
rotate 1 steps to the right: [99,-1,-100,3]
rotate 2 steps to the right: [3,99,-1,-100]

Note:
Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.
Could you do it in-place with O(1) extra space?
"""

class Solution:
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        arr = [0] * len(nums)
        
        for i in range(len(nums)):
            arr[(i + k) % len(nums)] = nums[i]
        
        for i in range(len(arr)):
            nums[i] = arr[i]

#Alternative O(1) space solution
class Solution:
    def rotate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) == 0 or k == 0: 
            return
        
        k = k % len(nums)
        count = 0
        start = 0
        
        while count < len(nums):
            current = start
            prev = nums[start]
            while True:
                nxt = (current + k) % len(nums)
                temp = nums[nxt]
                nums[nxt] = prev
                prev = temp
                current = nxt
                count += 1
                if current == start:
                    break
            start += 1

"""
665. Non-decreasing Array

Given an array with n integers, your task is to check if it could become non-decreasing by modifying at most 
1 element.

We define an array is non-decreasing if array[i] <= array[i + 1] holds for every i (1 <= i < n).

Example 1:
Input: [4,2,3]
Output: True
Explanation: You could modify the first 4 to 1 to get a non-decreasing array.

Example 2:
Input: [4,2,1]
Output: False
Explanation: You can't get a non-decreasing array by modify at most one element.

Note: The n belongs to [1, 10,000].
"""

class Solution:
    def checkPossibility(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) <= 0: 
            return False
        
        min_num = nums[0]
        larger_found = False

        for i in range(1, len(nums)):
            if nums[i] < min_num:
                if larger_found: 
                    return False
                if i - 2 < 0 or nums[i] > nums[i-2]:
                    min_num = nums[i]
                larger_found = True
            else: 
                min_num = nums[i]
                
        return True