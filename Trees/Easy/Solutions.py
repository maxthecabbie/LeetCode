"""
617. Merge Two Binary Trees

Given two binary trees and imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not.

You need to merge them into a new binary tree. The merge rule is that if two nodes overlap, then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of new tree.

Example 1:
Input: 
    Tree 1                     Tree 2                  
          1                         2                             
         / \                       / \                            
        3   2                     1   3                        
       /                           \   \                      
      5                             4   7                  
Output: 
Merged tree:
         3
        / \
       4   5
      / \   \ 
     5   4   7

Note: The merging process must start from the root nodes of both trees.
"""
class Solution:
    def mergeTrees(self, t1, t2):
        """
        :type t1: TreeNode
        :type t2: TreeNode
        :rtype: TreeNode
        """
        if t1 is None: 
            return t2
        if t2 is None: 
            return t1
        
        t1.val += t2.val
        t1.left = self.mergeTrees(t1.left, t2.left)
        t1.right = self.mergeTrees(t1.right, t2.right)
        
        return t1

"""
104. Maximum Depth of Binary Tree

Given a binary tree, find its maximum depth.

The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

Note: A leaf is a node with no children.

Example:
Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7
return its depth = 3.
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None: 
            return 0
        return max(1 + self.maxDepth(root.left), 1 + self.maxDepth(root.right))

"""
226. Invert Binary Tree

Invert a binary tree.

Example:
Input:

     4
   /   \
  2     7
 / \   / \
1   3 6   9
Output:

     4
   /   \
  7     2
 / \   / \
9   6 3   1
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def invertTree(self, root):
        """
        :type root: TreeNode
        :rtype: TreeNode
        """
        if root is None: 
            return None

        temp = root.left
        root.left = self.invertTree(root.right)
        root.right = self.invertTree(temp)
        return root

"""
653. Two Sum IV - Input is a BST

Given a Binary Search Tree and a target number, return true if there exist two elements in the BST such that their sum is equal to the given target.

Example 1:
Input: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 9

Output: True

Example 2:
Input: 
    5
   / \
  3   6
 / \   \
2   4   7

Target = 28

Output: False
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        return self.helper(root, k, set())
    
    def helper(self, root, k, num_set):
        if root is None: 
            return False
        
        target = k - root.val
        
        if target in num_set: 
            return True
        num_set.add(root.val)
        
        left = self.helper(root.left, k, num_set)
        right = self.helper(root.right, k, num_set)
        return left or right

"""
108. Convert Sorted Array to Binary Search Tree

Given an array where elements are sorted in ascending order, convert it to a height balanced BST.

For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

Example:

Given the sorted array: [-10,-3,0,5,9],

One possible answer is: [0,-3,9,-10,null,5], which represents the following height balanced BST:

      0
     / \
   -3   9
   /   /
 -10  5
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        return self.helper(nums, 0, len(nums) - 1)
    
    def helper(self, arr, lo, hi):
        if lo > hi:
            return None
        
        mid = int((lo + hi)/2)
        new_node = TreeNode(arr[mid])
        new_node.left = self.helper(arr, lo, mid-1)
        new_node.right = self.helper(arr, mid+1, hi)
        
        return new_node

"""
270. Closest Binary Search Tree Value

Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.

Note:
Given target value is a floating point.
You are guaranteed to have only one unique value in the BST that is closest to the target.
Example:

Input: root = [4,2,5,1,3], target = 3.714286

    4
   / \
  2   5
 / \
1   3

Output: 4
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def closestValue(self, root, target):
        """
        :type root: TreeNode
        :type target: float
        :rtype: int
        """
        closest = root.val
        
        while root is not None:
            if abs(target - root.val) < abs(target - closest):
                closest = root.val
                
            if target <= root.val:
                root = root.left
            else:
                root = root.right
                
        return closest

"""
101. Symmetric Tree

Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

    1
   / \
  2   2
 / \ / \
3  4 4  3

But the following [1,2,2,null,3,null,3] is not:
    1
   / \
  2   2
   \   \
   3    3

Note:
Bonus points if you could solve it both recursively and iteratively.
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def isSymmetric(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.helper(root, root)
    
    def helper(self, r1, r2):
        if r1 is None and r2 is None: 
            return True
        elif r1 is None or r2 is None: 
            return False
        elif r1.val != r2.val: 
            return False
        
        return self.helper(r1.left, r2.right) and self.helper(r1.right, r2.left)

"""
572. Subtree of Another Tree

Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

Example 1:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4 
  / \
 1   2
Return true, because t has the same structure and node values with a subtree of s.

Example 2:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
    /
   0
Given tree t:
   4
  / \
 1   2
Return false.
"""

class Solution:
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        if s is None: 
            return False
        
        if s.val == t.val:
            if self.is_equal(s, t): 
                return True

        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)
    
    def is_equal(self, s, t):
        if s is None and t is None: 
            return True
        elif s is None or t is None: 
            return False
        
        return s.val == t.val and self.is_equal(s.left, t.left) \
        and self.is_equal(s.right, t.right)
