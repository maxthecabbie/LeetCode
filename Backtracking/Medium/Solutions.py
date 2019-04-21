"""
39. Combination Sum

Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), find all
unique combinations in candidates where the candidate numbers sums to target.

The same repeated number may be chosen from candidates unlimited number of times.

Note:
1. All numbers (including target) will be positive integers.
2. The solution set must not contain duplicate combinations.

Example 1:
Input: candidates = [2,3,6,7], target = 7,
A solution set is:
[
  [7],
  [2,2,3]
]

Example 2:
Input: candidates = [2,3,5], target = 8,
A solution set is:
[
  [2,2,2,2],
  [2,3,3],
  [3,5]
]
"""

class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        
        def combination_sum_impl(start, target, curr):
            if target == 0:
                res.append(curr)
                return
            
            for i in range(start, len(candidates)):
                count = 1
                
                while count * candidates[i] <= target:
                    copy = curr[:]
                    for _ in range(count):
                        copy.append(candidates[i])
                        
                    combination_sum_impl(i+1, target - (count * candidates[i]), copy)
                    count += 1
        
        candidates.sort()
        combination_sum_impl(0, target, [])
        
        return res