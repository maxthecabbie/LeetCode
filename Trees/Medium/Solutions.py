"""
938. Range Sum of BST

Given the root node of a binary search tree, return the sum of values of all nodes with value between L and R
(inclusive).

The binary search tree is guaranteed to have unique values.

Example 1:
Input: root = [10,5,15,3,7,null,18], L = 7, R = 15
Output: 32

Example 2:
Input: root = [10,5,15,3,7,13,18,1,null,6], L = 6, R = 10
Output: 23
 
Note:
1. The number of nodes in the tree is at most 10000.
2. The final answer is guaranteed to be less than 2^31.
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def rangeSumBST(self, root, L, R):
        """
        :type root: TreeNode
        :type L: int
        :type R: int
        :rtype: int
        """
        if root is None:
            return 0
        
        total = root.val if L <= root.val <= R else 0
        
        if L < root.val:
            total += self.rangeSumBST(root.left, L, R)
        if R > root.val:
            total += self.rangeSumBST(root.right, L, R)
        
        return total

"""
979. Distribute Coins in Binary Tree

Given the root of a binary tree with N nodes, each node in the tree has node.val coins, and there are N coins
total.

In one move, we may choose two adjacent nodes and move one coin from one node to another.  (The move may be
from parent to child, or from child to parent.)

Return the number of moves required to make every node have exactly one coin.

https://leetcode.com/problems/distribute-coins-in-binary-tree/

Example 1:
Input: [3,0,0]
Output: 2
Explanation: From the root of the tree, we move one coin to its left child, and one coin to its right child.

Example 2:
Input: [0,3,0]
Output: 3
Explanation: From the left child of the root, we move two coins to the root [taking two moves].  Then, we
move one coin from the root of the tree to the right child.

Example 3:
Input: [1,0,2]
Output: 2

Example 4:
Input: [1,0,0,null,3]
Output: 4

Note:
1<= N <= 100
0 <= node.val <= N
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def distributeCoins(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.distribute_coins_impl(root)[1]
    
    def distribute_coins_impl(self, root):
        if root is None:
            return (0, 0)

        left = self.distribute_coins_impl(root.left)
        right = self.distribute_coins_impl(root.right)
        
        child_coins = left[0] + right[0]
        moves = left[1] + right[1]
        
        coins = child_coins - (root.val - 1)
        if coins == 0:
            return (0, moves)
        elif coins < 0:
            give = (root.val - 1) - child_coins
            moves += give
            return (-give, moves)
        else:
            need = child_coins - (root.val - 1)
            moves += need
            return (need, moves)
        

"""
729. My Calendar I

Implement a MyCalendar class to store your events. A new event can be added if adding the event will not 
cause a double booking.

Your class will have the method, book(int start, int end). Formally, this represents a booking on the half 
open interval [start, end), the range of real numbers x such that start <= x < end.

A double booking happens when two events have some non-empty intersection (ie., there is some time that is 
common to both events.)

For each call to the method MyCalendar.book, return true if the event can be added to the calendar 
successfully without causing a double booking. Otherwise, return false and do not add the event to the 
calendar.

Your class will be called like this: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)

Example 1:
MyCalendar();
MyCalendar.book(10, 20); // returns true
MyCalendar.book(15, 25); // returns false
MyCalendar.book(20, 30); // returns true
Explanation: 
The first event can be booked.  The second can't because time 15 is already booked by another event.
The third event can be booked, as the first event takes every time less than 20, but not including 20.

Note:
The number of calls to MyCalendar.book per test case will be at most 1000.
In calls to MyCalendar.book(start, end), start and end are integers in the range [0, 10^9].
"""

class Node:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.left = self.right = None
    
    def insert(self, node):
        if node.end <= self.start:
            if self.left:
                return self.left.insert(node)
            else:
                self.left = node
                return True
        elif node.start >= self.end:
            if self.right:
                return self.right.insert(node)
            else:
                self.right = node
                return True
        else:
            return False
            
class MyCalendar:
    def __init__(self):
        self.root = None

    def book(self, start, end):
        """
        :type start: int
        :type end: int
        :rtype: bool
        """
        new_node = Node(start, end)
        if self.root is None:
            self.root = new_node
            return True
        else:
            return self.root.insert(new_node)

# Your MyCalendar object will be instantiated and called as such:
# obj = MyCalendar()
# param_1 = obj.book(start,end)

"""
314. Binary Tree Vertical Order Traversal

Given a binary tree, return the vertical order traversal of its nodes' values. (ie, from top to bottom, 
column by column).

If two nodes are in the same row and column, the order should be from left to right.

Examples 1:
Input: [3,9,20,null,null,15,7]

   3
  /\
 /  \
 9  20
    /\
   /  \
  15   7 

Output:

[
  [9],
  [3,15],
  [20],
  [7]
]

Examples 2:
Input: [3,9,8,4,0,1,7]

     3
    /\
   /  \
   9   8
  /\  /\
 /  \/  \
 4  01   7 

Output:

[
  [4],
  [9],
  [3,0,1],
  [8],
  [7]
]

Examples 3:
Input: [3,9,8,4,0,1,7,null,null,null,2,5] (0's right child is 2 and 1's left child is 5)

     3
    /\
   /  \
   9   8
  /\  /\
 /  \/  \
 4  01   7
    /\
   /  \
   5   2

Output:

[
  [4],
  [9,5],
  [3,0,1],
  [8,2],
  [7]
]
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def verticalOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root is None:
            return []
        
        levels = {}
        min_level = 0
        curr_layer = [[0, root]]
        
        while curr_layer:
            next_layer = []
            
            for entry in curr_layer:
                level = entry[0]
                node = entry[1]
                levels.setdefault(level, []).append(node.val)
                min_level = min(min_level, level)
                
                if node.left:
                    next_layer.append([level - 1, node.left])
                if node.right:
                    next_layer.append([level + 1, node.right])
            curr_layer = next_layer
        
        result = []
        while levels.get(min_level, None):
            result.append(levels[min_level])
            min_level += 1
        return result

"""
103. Binary Tree Zigzag Level Order Traversal

Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, 
then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],

    3
   / \
  9  20
    /  \
   15   7

return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        curr = [root] if root else []
        res = []
        while curr:
            next_level = []
            self.append_curr(curr, res)
            for i in range(len(curr)):
                node = curr[i]
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
            curr = next_level
        
        return res
    
    def append_curr(self, curr, res):
        lev = []
        s = 0 if len(res) % 2 == 0 else len(curr) - 1
        e = -1 if len(res) % 2 == 1 else len(curr)
        d = 1 if s < e else -1
        for i in range(s, e, d):
            lev.append(curr[i].val)
        res.append(lev)

"""
105. Construct Binary Tree from Preorder and Inorder Traversal

Given preorder and inorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given

preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
Return the following binary tree:
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """
        inorder_map = {inorder[i] : i for i in range(len(inorder))}
        preorder.reverse()
        return self.build_tree_impl(preorder, inorder_map, 0, len(inorder) - 1)
    
    def build_tree_impl(self, preorder, inorder_map, left, right):
        if left <= right:
            root = TreeNode(preorder.pop())
            root_idx = inorder_map[root.val]
            
            root.left = self.build_tree_impl(preorder, inorder_map, left, root_idx - 1)
            root.right = self.build_tree_impl(preorder, inorder_map, root_idx + 1, right)
            
            return root


"""
450. Delete Node in a BST

Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the 
root node reference (possibly updated) of the BST.

Basically, the deletion can be divided into two stages:

Search for a node to remove.
If the node is found, delete the node.
Note: Time complexity should be O(height of tree).

Example:
root = [5,3,6,2,4,null,7]
key = 3

    5
   / \
  3   6
 / \   \
2   4   7

Given key to delete is 3. So we find the node with value 3 and delete it.

One valid answer is [5,4,6,2,null,null,7], shown in the following BST.

    5
   / \
  4   6
 /     \
2       7

Another valid answer is [5,2,6,null,4,null,7].

    5
   / \
  2   6
   \   \
    4   7
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def deleteNode(self, root, key):
        """
        :type root: TreeNode
        :type key: int
        :rtype: TreeNode
        """
        if root is None:
            return None
        
        if key == root.val:
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            
            temp = root.right
            while temp.left:
                temp = temp.left
            root.val = temp.val
            root.right = self.deleteNode(root.right, temp.val)
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        else :
            root.right = self.deleteNode(root.right, key)
        
        return root

"""
654. Maximum Binary Tree

Given an integer array with no duplicates. A maximum tree building on this array is defined as follow:

The root is the maximum number in the array.
The left subtree is the maximum tree constructed from left part subarray divided by the maximum number.
The right subtree is the maximum tree constructed from right part subarray divided by the maximum number.
Construct the maximum tree by the given array and output the root node of this tree.

Example 1:
Input: [3,2,1,6,0,5]
Output: return the tree root node representing the following tree:

      6
    /   \
   3     5
    \    / 
     2  0   
       \
        1

Note:
The size of the given array will be in the range [1,1000].
"""

class Solution:
    def constructMaximumBinaryTree(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        stack = []

        for i in range(len(nums)):
            node = TreeNode(nums[i])
            while len(stack) > 0 and node.val > stack[-1].val:
                node.left = stack.pop()

            if len(stack) > 0:
                stack[-1].right = node
            stack.append(node)

        return None if len(stack) == 0 else stack[0]

"""
94. Binary Tree Inorder Traversal


Given a binary tree, return the inorder traversal of its nodes' values.

Example:
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,3,2]

Follow up: Recursive solution is trivial, could you do it iteratively?
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root is None:
            return []
        
        cursor = root.left
        stack = [root]
        res = []
        
        while stack or cursor:
            while cursor:
                stack.append(cursor)
                cursor = cursor.left
            
            cursor = stack.pop()
            res.append(cursor.val)
            cursor = cursor.right
        
        return res

"""
144. Binary Tree Preorder Traversal

Given a binary tree, return the preorder traversal of its nodes' values.

Example:
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [1,2,3]

Follow up: Recursive solution is trivial, could you do it iteratively?

"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        nodes = []
        self.preorder(root, nodes)
        return nodes
    
    def preorder(self, root, nodes):
        if root is None:
            return
        nodes.append(root.val)
        self.preorder(root.left, nodes)
        self.preorder(root.right, nodes)

# Alternative solution using a stack
class Solution(object):
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root is None:
            return []
        
        nodes = []
        stack = [root]
        
        while stack:
            node = stack.pop()
            nodes.append(node.val)
            if node.right:
                stack.append(node.right)
            if node.left:
                stack.append(node.left)
        
        return nodes

"""
230. Kth Smallest Element in a BST

Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note: 
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

Example 1:
Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1

Example 2:
Input: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
Output: 3

Follow up:
What if the BST is modified (insert/delete operations) often and you need to find the kth smallest 
frequently? How would you optimize the kthSmallest routine?

Solution Explanation
- Start from root and keep going left pushing all nodes onto a stack
- Then for k-1 times, pop a node off the stack
    - If it has a right child then add that to the stack and keep going left from the right child, pushing 
    all those nodes on the stack. This is essentially in order traversal
- Finish by popping the last pushed node off stack and returning the val
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        if root is None:
            return None
        
        stack = []
        while root:
            stack.append(root)
            root = root.left
        
        for _ in range(k-1):
            node = stack.pop()
            node = node.right
            while node:
                stack.append(node)
                node = node.left

        return stack.pop().val

"""
173. Binary Search Tree Iterator

Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node 
of a BST.

Calling next() will return the next smallest number in the BST.

Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of 
the tree.
"""

# Definition for a  binary tree node
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.stack = []

        while root:
            self.stack.append(root)
            root = root.left

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.stack) > 0

    def next(self):
        """
        :rtype: int
        """
        node = self.stack.pop()
        cursor = node.right

        while cursor:
            self.stack.append(cursor)
            cursor = cursor.left
        return node.val

# Your BSTIterator will be called like this:
# i, v = BSTIterator(root), []
# while i.hasNext(): v.append(i.next())

"""
102. Binary Tree Level Order Traversal

Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level 
by level).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its level order traversal as:
[
  [3],
  [9,20],
  [15,7]
]
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if root is None:
            return []
        
        result = []
        curr_layer = [root]
        
        while curr_layer:
            new_layer = []
            node_vals = []
            
            for node in curr_layer:
                node_vals.append(node.val)
                
                if node.left:
                    new_layer.append(node.left)
                if node.right:
                    new_layer.append(node.right)
                    
            curr_layer = new_layer
            result.append(node_vals)
        
        return result

"""
549. Binary Tree Longest Consecutive Sequence II

Given a binary tree, you need to find the length of Longest Consecutive Path in Binary Tree.

Especially, this path can be either increasing or decreasing. For example, [1,2,3,4] and [4,3,2,1] are both 
considered valid, but the path [1,2,4,3] is not valid. On the other hand, the path can be in the 
child-Parent-child order, where not necessarily be parent-child order.

Example 1:
Input:
        1
       / \
      2   3
Output: 2
Explanation: The longest consecutive path is [1, 2] or [2, 1].

Example 2:
Input:
        2
       / \
      1   3
Output: 3
Explanation: The longest consecutive path is [1, 2, 3] or [3, 2, 1].

Note: All the values of tree nodes are in the range of [-1e7, 1e7].
"""

class Solution:
    def longestConsecutive(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        max_n = [0]
        self.longest_helper(root, max_n)
        return max_n[0]

    def longest_helper(self, root, max_n):
        if root is None:
            return (0, 0)

        inc, dcr = 1, 1
        if root.left:
            l = self.longest_helper(root.left, max_n)
            if root.val - root.left.val == 1:
                inc += l[0]
            elif root.left.val - root.val == 1:
                dcr += l[1]

        if root.right:
            r = self.longest_helper(root.right, max_n)
            if root.val - root.right.val == 1:
                inc = max(inc, r[0] + 1)
            elif root.right.val - root.val == 1:
                dcr = max(dcr, r[1] + 1)

        max_n[0] = max(max_n[0], inc + dcr - 1)
        return (inc, dcr)

"""
298. Binary Tree Longest Consecutive Sequence

Given a binary tree, find the length of the longest consecutive sequence path.

The path refers to any sequence of nodes from some starting node to any node in the tree along the 
parent-child connections. The longest consecutive path need to be from parent to child (cannot be the 
reverse).

Example 1:

Input:

   1
    \
     3
    / \
   2   4
        \
         5

Output: 3
Explanation: Longest consecutive sequence path is 3-4-5, so return 3.

Example 2:

Input:

   2
    \
     3
    / 
   2    
  / 
 1

Output: 2 
Explanation: Longest consecutive sequence path is 2-3, not 3-2-1, so return 2.
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def longestConsecutive(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        return self.find_longest(root, root.val, -sys.maxsize - 1)
        
    def find_longest(self, root, prev, n):
        if root is None: 
            return n
        
        next_n = n + 1
        if root.val - prev != 1:
            next_n = 1
        
        left = self.find_longest(root.left, root.val, next_n)
        right = self.find_longest(root.right, root.val, next_n)
        return max(n, max(left, right))

"""
129. Sum Root to Leaf Numbers

Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.

An example is the root-to-leaf path 1->2->3 which represents the number 123.

Find the total sum of all root-to-leaf numbers.

Note: A leaf is a node with no children.

Example:
Input: [1,2,3]
    1
   / \
  2   3
Output: 25
Explanation:
The root-to-leaf path 1->2 represents the number 12.
The root-to-leaf path 1->3 represents the number 13.
Therefore, sum = 12 + 13 = 25.

Example 2:

Input: [4,9,0,5,1]
    4
   / \
  9   0
 / \
5   1
Output: 1026
Explanation:
The root-to-leaf path 4->9->5 represents the number 495.
The root-to-leaf path 4->9->1 represents the number 491.
The root-to-leaf path 4->0 represents the number 40.
Therefore, sum = 495 + 491 + 40 = 1026.

Solution Explanation
- Starting from the root we go to the left and right nodes
    - If the root is None, we return 0
    - Else if both the left and right nodes are None, then we must return the n * 10 + root.val
    - Else, traverse to the left and right nodes passing in n, which has been updated to n * 10 + root.val 
    and return the sum of the result from the left and right nodes
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        return self.sum_helper(root, 0)
    
    def sum_helper(self, root, n):
        if root is None:
            return 0
        
        n = n * 10 + root.val
        if not root.left and not root.right:
            return n
        
        return self.sum_helper(root.left, n) + self.sum_helper(root.right, n)

"""
331. Verify Preorder Serialization of a Binary Tree

One way to serialize a binary tree is to use pre-order traversal. When we encounter a non-null node, we 
record the node's value. If it is a null node, we record using a sentinel value such as #.

     _9_
    /   \
   3     2
  / \   / \
 4   1  #  6
/ \ / \   / \
# # # #   # #

For example, the above binary tree can be serialized to the string "9,3,4,#,#,1,#,#,2,#,6,#,#", where # 
represents a null node.

Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of 
a binary tree. Find an algorithm without reconstructing the tree.

Each comma separated value in the string must be either an integer or a character '#' representing null 
pointer.

You may assume that the input format is always valid, for example it could never contain two consecutive 
commas such as "1,,3".

Example 1:
Input: "9,3,4,#,#,1,#,#,2,#,6,#,#"
Output: true

Example 2:
Input: "1,#"
Output: false

Example 3:
Input: "9,#,#,1"
Output: false

- Using a stack we loop through the nodes
    - If a node is a number, we add to stack
    - If node is #:
        - If top element on stack is number, add # to stack
        - If top element is #, we pop the # off the stack and check if the stack is empty
            - If stack is empty, we return False since we know that the preorder traversal is not valid
            - If stack is not empty, we pop the next element off the stack which is a number
            - If the next element in stack is another #, repeat the above two steps until the top element is 
            not a #
            - Finally add a # onto the stack
- Return whether the stack length is 1 and the top element is a #
- NOTE this works because when we encounter a #, if the top element in the stack is number, we know that 
this # is on the left subtree of that number. So we can just add # to the stack
- If the top element is a #, that means this # is the right subtree of the next number in the stack. In this 
case we pop the # and the number off the stack. We keep doing this until the top level element is not a #. 
The logic behind this is that two # elements indicates we have finished traversing the next number in the 
stack. We can "replace" this number with a # to indicate that. But if we get another # on the top of the 
stack again, we know we have finished traversing the next number in the stack as well. So that is why we 
keep going until the top element is not a #
- In the end once we are finished, since we put a # when we are done with an element, there should be one # 
left in the stack and we check for this case
"""

class Solution:
    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """
        s = []
        preorder = preorder.split(",")
        
        for c in preorder:
            if c.isdigit():
                s.append(c)
                
            elif c == "#":
                while len(s) > 0 and s[-1] == "#":
                    s.pop()
                    if len(s) <= 0:
                        return False
                    s.pop()
                s.append("#")
            
        return len(s) == 1 and s[-1] == "#"

# Alternative solution using indegree and outdegree
"""
- Keep track of the outdegree and indegree
- outdegree is the number of edges going out from a node and indegree is number of edges coming in to a node
- Difference between out_degree and in_degree should never be negative
- Every node we "visit" we increase in_degree by 1, except the first node since it has no parent
- For every non # node, we increase out_degree by 2
- In the end, check that out_degree and in_degree are the same
"""
class Solution:
    def isValidSerialization(self, preorder):
        """
        :type preorder: str
        :rtype: bool
        """        
        preorder = preorder.split(",")
        out_degree = 0
        in_degree = 0
        
        for i in range(len(preorder)):
            if i != 0:
                in_degree += 1
            if out_degree - in_degree < 0:
                return False
            if preorder[i] != "#":
                out_degree += 2
                
        return out_degree == in_degree

"""
261. Graph Valid Tree

Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a 
function to check whether these edges make up a valid tree.

Example 1:
Input: n = 5, and edges = [[0,1], [0,2], [0,3], [1,4]]
Output: true
Example 2:

Input: n = 5, and edges = [[0,1], [1,2], [2,3], [1,3], [1,4]]
Output: false

Note: you can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0,1] is 
the same as [1,0] and thus will not appear together in edges.
"""

class Solution:
    def validTree(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: bool
        """
        graph = {}
        visited = set()
        
        for e in edges:
            graph.setdefault(e[0], set()).add(e[1])
            graph.setdefault(e[1], set()).add(e[0])
        
        self.dfs(0, graph, visited)
        return len(visited) == n and len(edges) == n - 1
    
    def dfs(self, node, graph, visited):
        visited.add(node)
        
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                self.dfs(neighbor, graph, visited)

"""
116. Populating Next Right Pointers in Each Node

Given a binary tree

struct TreeLinkNode {
  TreeLinkNode *left;
  TreeLinkNode *right;
  TreeLinkNode *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer 
should be set to NULL.

Initially, all next pointers are set to NULL.

Note:
You may only use constant extra space.
Recursive approach is fine, implicit stack space does not count as extra space for this problem.
You may assume that it is a perfect binary tree (ie, all leaves are at the same level, and every parent has 
two children).

Example:
Given the following perfect binary tree,

     1
   /  \
  2    3
 / \  / \
4  5  6  7
After calling your function, the tree should look like:

     1 -> NULL
   /  \
  2 -> 3 -> NULL
 / \  / \
4->5->6->7 -> NULL
"""

# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None
class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        curr = [root] if root else []
        
        while curr:
            next_level = []
            
            for i in range(len(curr) - 1):
                curr[i].next = curr[i+1]
            
            for node in curr:
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
                    
            curr = next_level

"""
285. Inorder Successor in BST

Given a binary search tree and a node in it, find the in-order successor of that node in the BST.

Note: If the given node has no in-order successor in the tree, return null.

Example 1:
Input: root = [2,1,3], p = 1

  2
 / \
1   3

Output: 2

Example 2:
Input: root = [5,3,6,2,4,null,null,1], p = 6

      5
     / \
    3   6
   / \
  2   4
 /   
1

Output: null
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def inorderSuccessor(self, root, p):
        """
        :type root: TreeNode
        :type p: TreeNode
        :rtype: TreeNode
        """
        if p.right:
            cursor = p.right
            while cursor.left:
                cursor = cursor.left
            return cursor
        
        parent = self.find_inorder_parent(root, p)
        return parent if parent is not p else None
    
    def find_inorder_parent(self, root, p):
        if root is p:
            return p
        
        if p.val > root.val:
            return self.find_inorder_parent(root.right, p)
        
        left = self.find_inorder_parent(root.left, p)
        return root if left is p else left

"""
222. Count Complete Tree Nodes

Given a complete binary tree, count the number of nodes.

Note:
Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the
last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.

Example:
Input: 
    1
   / \
  2   3
 / \  /
4  5 6

Output: 6
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
            return 0
        
        left_d = self.depth(root.left)
        right_d = self.depth(root.right)
        
        if left_d == right_d:
            return 2**left_d + self.countNodes(root.right)
        else:
            return 2**right_d + self.countNodes(root.left)
    
    def depth(self, root):
        if not root:
            return 0
        return 1 + self.depth(root.left)
  
"""
236. Lowest Common Ancestor of a Binary Tree

Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p 
and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant 
of itself).”

Given the following binary tree:  root = [3,5,1,6,2,0,8,null,null,7,4]

        _______3______
       /              \
    ___5__          ___1__
   /      \        /      \
   6      _2       0       8
         /  \
         7   4
         
Example 1:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of of nodes 5 and 1 is 3.

Example 2:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: The LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself
             according to the LCA definition.

Note:
All of the nodes' values will be unique.
p and q are different and both values will exist in the binary tree.
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if root is None:
            return None
        
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        if left and right:
            return root
        
        if root is p or root is q:
            return root

        return left if left else right

"""
310. Minimum Height Trees

For a undirected graph with tree characteristics, we can choose any node as the root. The result graph is 
then a rooted tree. Among all possible rooted trees, those with minimum height are called minimum height 
trees (MHTs). Given such a graph, write a function to find all the MHTs and return a list of their root 
labels.

Format
The graph contains n nodes which are labeled from 0 to n - 1. You will be given the number n and a list of 
undirected edges (each edge is a pair of labels).

You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the 
same as [1, 0] and thus will not appear together in edges.

Example 1:
Input: n = 4, edges = [[1, 0], [1, 2], [1, 3]]

        0
        |
        1
       / \
      2   3 

Output: [1]

Example 2:
Input: n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]

     0  1  2
      \ | /
        3
        |
        4
        |
        5 

Output: [3, 4]

Note:
According to the definition of tree on Wikipedia: “a tree is an undirected graph in which any two vertices 
are connected by exactly one path. In other words, any connected graph without simple cycles is a tree.”
The height of a rooted tree is the number of edges on the longest downward path between the root and a leaf.
"""

class Solution:
    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        graph = {}
        nodes = {i for i in range(n)}
        
        for e in edges:
            graph.setdefault(e[0], set()).add(e[1])
            graph.setdefault(e[1], set()).add(e[0])
        leaves = [key for key in graph if len(graph[key]) == 1]
        
        while len(nodes) > 2:
            new_leaves = []
            
            for node in leaves:
                nodes.remove(node)
                parent = graph[node].pop()
                graph[parent].remove(node)
                
                if len(graph[parent]) == 1:
                    new_leaves.append(parent)
            leaves = new_leaves
        
        result = [node for node in nodes]
        return sorted(result)

"""
98. Validate Binary Search Tree

Given a binary tree, determine if it is a valid binary search tree (BST).

Assume a BST is defined as follows:

The left subtree of a node contains only nodes with keys less than the node's key.
The right subtree of a node contains only nodes with keys greater than the node's key.
Both the left and right subtrees must also be binary search trees.

Example 1:
Input:
    2
   / \
  1   3
Output: true

Example 2:
    5
   / \
  1   4
     / \
    3   6
Output: false
Explanation: The input is: [5,1,4,null,null,3,6]. The root node's value
             is 5 but its right child's value is 4.

Solution Explanation
- Checking each child of a node individually will not work
- Must have a min_val and max_val that you update as you go to the left and right children
- When you go left, update the max_val to be the val of the node you are at
- When you go right, update the min_val to the val of the node you are at
- At each node, check to see if the val of the node is between the min_val and max_val
    - If yes, keep going
    - If no, return False
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution:
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.valid_helper(root, None, None)
    
    def valid_helper(self, root, min_val, max_val):
        if root is None:
            return True
        
        if min_val is not None and root.val <= min_val:
            return False
        if max_val is not None and root.val >= max_val:
            return False
        
        return self.valid_helper(root.left, min_val, root.val) and\
        self.valid_helper(root.right, root.val, max_val)