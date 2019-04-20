"""
728. Self Dividing Numbers

A self-dividing number is a number that is divisible by every digit it contains.

For example, 128 is a self-dividing number because 128 % 1 == 0, 128 % 2 == 0, and 128 % 8 == 0.

Also, a self-dividing number is not allowed to contain the digit zero.

Given a lower and upper number bound, output a list of every possible self dividing number, including the 
bounds if possible.

Example 1:
Input: 
left = 1, right = 22
Output: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 15, 22]

Note:
The boundaries of each input argument are 1 <= left <= right <= 10000.
"""
class Solution:
    def selfDividingNumbers(self, left, right):
        """
        :type left: int
        :type right: int
        :rtype: List[int]
        """
        result = []
        for i in range(left, right+1):
            if self.self_dividing(i):
                result.append(i)
        return result
        
    def self_dividing(self, n):
        n_cpy = n
        while n_cpy > 0:
            dig = n_cpy % 10
            if dig == 0 or n % dig != 0:
                return False
            n_cpy = n_cpy//10
        return True

"""
202. Happy Number

Write an algorithm to determine if a number is "happy".

A happy number is a number defined by the following process: Starting with any positive integer, replace the
number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it
will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process
ends in 1 are happy numbers.

Example: 
Input: 19
Output: true
Explanation: 
12 + 92 = 82
82 + 22 = 68
62 + 82 = 100
12 + 02 + 02 = 1
"""
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        cache = set()
        
        while n != 1:
            if n in cache:
                return False
            cache.add(n)
            
            digits = []
            pos = 1
            
            while n // pos > 0:
                digits.append((n // pos) % 10)
                pos *= 10
            
            n = sum([digit*digit for digit in digits])
            
        return True

"""
204. Count Primes

Count the number of prime numbers less than a non-negative number, n.

Example:
Input: 10
Output: 4
Explanation: There are 4 prime numbers less than 10, they are 2, 3, 5, 7.
"""
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 1:
            return 0
        
        primes = [True for _ in range(n)]
        primes[0], primes[1] = False, False
        
        i = 2
        while i*i <= n:
            if primes[i]:
                j = i*i
                while j < n:
                    primes[j] = False
                    j += i
            i += 1
    
        count = 0
        for n in primes:
            if n:
                count += 1
        
        return count

"""
7. Reverse Integer

Given a 32-bit signed integer, reverse digits of an integer.

Example 1:
Input: 123
Output: 321

Example 2:
Input: -123
Output: -321

Example 3:
Input: 120
Output: 21

Note:
Assume we are dealing with an environment which could only store integers within the 32-bit signed integer 
range: [−231,  231 − 1]. For the purpose of this problem, assume that your function returns 0 when the 
reversed integer overflows.
"""
class Solution:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        sign = -1 if x < 0 else 1
        x = abs(x)
        n = 0
        
        while x > 0:
            digit = x % 10
            n = n * 10 + digit
            x = int(x/10)
            
        if n > 2**31 - 1 or n < -2**31: 
            return 0
        return n * sign
        