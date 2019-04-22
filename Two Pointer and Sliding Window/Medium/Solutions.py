"""
16. 3Sum Closest

Given an array nums of n integers and an integer target, find three integers in nums such that the sum is
closest to target. Return the sum of the three integers. You may assume that each input would have exactly
one solution.

Example:
Given array nums = [-1, 2, 1, -4], and target = 1.

The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
"""

class Solution(object):
    def threeSumClosest(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        closest = nums[0] + nums[1] + nums[2]
        
        nums.sort()
        for i in range(len(nums) - 2):
            k,j = i+1, len(nums) - 1
            
            while k < j:
                s = nums[i] + nums[k] + nums[j]
                
                if abs(target - s) < abs(target - closest):
                    closest = s
                
                if s <= target:
                    k += 1
                else:
                    j -= 1
            
        return closest
                
"""
904. Fruit Into Baskets

In a row of trees, the i-th tree produces fruit with type tree[i].

You start at any tree of your choice, then repeatedly perform the following steps:

Add one piece of fruit from this tree to your baskets.  If you cannot, stop.
Move to the next tree to the right of the current tree.  If there is no tree to the right, stop.
Note that you do not have any choice after the initial choice of starting tree: you must perform step 1, then
step 2, then back to step 1, then step 2, and so on until you stop.

You have two baskets, and each basket can hcarry any quantity of fruit, but you want each basket to only carry
one type of fruit each.

What is the total amount of fruit you can collect with this procedure?

Example 1:
Input: [1,2,1]
Output: 3

Explanation: We can collect [1,2,1].

Example 2:
Input: [0,1,2,2]
Output: 3

Explanation: We can collect [1,2,2].
If we started at the first tree, we would only collect [0, 1].

Example 3:
Input: [1,2,3,2,2]
Output: 4

Explanation: We can collect [2,3,2,2].
If we started at the first tree, we would only collect [1, 2].

Example 4:
Input: [3,3,3,1,2,1,1,2,3,3,4]
Output: 5

Explanation: We can collect [1,2,1,1,2].
If we started at the first tree or the eighth tree, we would only collect 4 fruits.
 
Note:
1 <= tree.length <= 40000
0 <= tree[i] < tree.length
"""

class Solution(object):
    def totalFruit(self, trees):
        max_len = 0
        
        l, r = 0, 0
        counts = {}

        while r < len(trees):
            counts[trees[r]] = counts.get(trees[r], 0) + 1

            while len(counts) > 2:
                counts[trees[l]] -= 1
                
                if counts[trees[l]] == 0:
                    del counts[trees[l]]
                l += 1

            max_len = max(max_len, r - l + 1)
            r += 1

        return max_len