"""
739. Daily Temperatures

Given a list of daily temperatures, produce a list that, for each day in the input, tells you how many days 
you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 
0 instead.

For example, given the list temperatures = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 
2, 1, 1, 0, 0].

Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the 
range [30, 100].
"""
class Solution:
    def dailyTemperatures(self, temperatures):
        """
        :type temperatures: List[int]
        :rtype: List[int]
        """
        stack = []
        result = [0 for _ in range(len(temperatures))]
        
        for i in range(len(temperatures)):
            temp = temperatures[i]
            while len(stack) > 0 and temp > stack[-1][0]:
                day = stack.pop()
                result[day[1]] = i - day[1]
            stack.append([temp, i])
        
        return result

"""
503. Next Greater Element II

Given a circular array (the next element of the last element is the first element of the array), print the 
Next Greater Number for every element. The Next Greater Number of a number x is the first greater number to 
its traversing-order next in the array, which means you could search circularly to find its next greater 
number. If it doesn't exist, output -1 for this number.

Example 1:
Input: [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2; 
The number 2 can't find next greater number; 
The second 1's next greater number needs to search circularly, which is also 2.

Note: The length of given array won't exceed 10000.
"""

class Solution:
    def nextGreaterElements(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = [-1 for _ in range(len(nums))]
        s = []
        
        for _ in range(2):
            for i in range(len(nums)):
                while len(s) > 0 and s[-1][0] < nums[i]:
                    waiting = s.pop()
                    res[waiting[1]] = nums[i]
                s.append([nums[i], i])
        return res

"""
341. Flatten Nested List Iterator

Given a nested list of integers, implement an iterator to flatten it.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Example 1:
Given the list [[1,1],2,[1,1]],

By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: 
[1,1,2,1,1].

Example 2:
Given the list [1,[4,[6]]],

By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: 
[1,4,6].

Solution Explanation
- For this question we use a stack, first in our init function we loop through the nestedList backwards and 
append to our stack and then call prep_next
- prep_next checks if the top item is a nested list and if it is, pops it off, gets all the items, loops 
through them backwards, and appends the items onto the stack. It keeps going until the top item is an integer
- next function simply returns the top item in the stack which should be an integer due to us calling 
prep_next in the init function and after every next call
- hasNext simply returns if the stack size is greater than 0
"""

# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """
class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.stack = []
        
        for i in range(len(nestedList) - 1, -1, -1):
            self.stack.append(nestedList[i])
        self.prep_next()

    def next(self):
        """
        :rtype: int
        """
        next_int = self.stack.pop().getInteger()
        
        if self.hasNext():
            self.prep_next()
        return next_int

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.stack) > 0
    
    def prep_next(self):
        while self.hasNext() and not self.stack[-1].isInteger():
            nested_list = self.stack.pop().getList()
            for i in range(len(nested_list) - 1, -1, -1):
                self.stack.append(nested_list[i])

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())

"""
394. Decode String

Given an encoded string, return it's decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being 
repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are 
well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for 
those repeat numbers, k. For example, there won't be input like 3a or 2[4].

Examples:
s = "3[a]2[bc]", return "aaabcbc".
s = "3[a2[c]]", return "accaccacc".
s = "2[abc]3[cd]ef", return "abcabccdcdcdef".

Solution Explanation
- Have two stacks, one for keeping track of numbers and the other for keeping track of strings
- Keep a buffer, which keeps track of the characters encountered so far
- When you encounter a number, push it onto the stack
- When you encounter "[", push the current buffer onto the strs stack and reset buffer to the empty string
- When you encounter "]", pop the last count from the mult stack and pop the last string from the strs stack 
and set temp equal to this string. If the count from the mult stack is n, append the buffer to temp n times 
and then set the buffer equal to temp
- When you encounter a character append it to the buffer
- Return the buffer joined as a string
"""

class Solution:
    def decodeString(self, s):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        mult = []
        strs = []
        buffer = [""]
        i = 0
        
        while i < len(s):
            if s[i].isdigit():
                count = 0
                while s[i].isdigit():
                    count = count * 10 + int(s[i])
                    i += 1
                mult.append(count)
            elif s[i] == "[":
                strs.append(buffer)
                buffer = [""]
                i += 1
            elif s[i] == "]":
                temp = strs.pop()
                count = mult.pop()
                for _ in range(count):
                    temp.append("".join(buffer))
                buffer = temp
                i += 1
            else:
                buffer.append(s[i])
                i += 1
                
        return "".join(buffer)

"""
150. Evaluate Reverse Polish Notation

Evaluate the value of an arithmetic expression in Reverse Polish Notation.

Valid operators are +, -, *, /. Each operand may be an integer or another expression.

Note:
Division between two integers should truncate toward zero.
The given RPN expression is always valid. That means the expression would always evaluate to a result and 
there won't be any divide by zero operation.

Example 1:
Input: ["2", "1", "+", "3", "*"]
Output: 9
Explanation: ((2 + 1) * 3) = 9

Example 2:
Input: ["4", "13", "5", "/", "+"]
Output: 6
Explanation: (4 + (13 / 5)) = 6

Example 3:
Input: ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
Output: 22
Explanation: 
  ((10 * (6 / ((9 + 3) * -11))) + 17) + 5
= ((10 * (6 / (12 * -11))) + 17) + 5
= ((10 * (6 / -132)) + 17) + 5
= ((10 * 0) + 17) + 5
= (0 + 17) + 5
= 17 + 5
= 22
"""

class Solution:
    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        
        for t in tokens:
            if t not in ["+", "-", "*", "/"]:
                stack.append(int(t))
            else:
                n2 = stack.pop()
                n1 = stack.pop()
                
                if t == "+":
                    n1 += n2
                elif t == "-":
                    n1 -= n2
                elif t == "/":
                    n1 = int(n1/n2)
                elif t == "*":
                    n1 *= n2
                stack.append(n1)
                
        return stack.pop() if len(stack) > 0 else 0

"""
402. Remove K Digits
- We are interested in numbers that are larger than the number to the right of it starting from the left
- Ideally we want to remove k of these numbers starting from the left, but what if there aren't k of these 
numbers?
- We have an array that functions as a stack. Iterate through our number and when we find a number that is 
smaller than the top number in the stack, we keep popping the numbers off the array as long as they are 
larger than the current number. We can do this a max of k times
- If we don't remove k numbers, then we remove the numbers from the end, since these are the largest. How 
many do we remove from the end? If we removed n numbers during the iteration phase, we remove k-n numbers 
from the end
- We must also remove leading zeros
- Return the number in the stack starting at the first non zero number. If there is no such number, return 0
"""

class Solution:
    def removeKdigits(self, num, k):
        """
        :type num: str
        :type k: int
        :rtype: str
        """
        stack = []
        for i in range(len(num)):
            c = num[i]
            while len(stack) > 0 and stack[-1] > c and k > 0:
                stack.pop()
                k -= 1
            stack.append(c)
            
        keep = len(stack) - k
        stack = stack[0:keep]
        i = 0
        
        while i < len(stack) and stack[i] == "0":
            i += 1
            
        return "0" if i == len(stack) else "".join(stack[i:])