"""
145. Binary Tree Postorder Traversal

Given a binary tree, return the postorder traversal of its nodes' values.

Example:
Input: [1,null,2,3]
   1
    \
     2
    /
   3

Output: [3,2,1]

Follow up: Recursive solution is trivial, could you do it iteratively?
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        nodes = []
        self.postorder(root, nodes)
        return nodes
    
    def postorder(self, root, nodes):
        if root is None:
            return
        
        self.postorder(root.left, nodes)
        self.postorder(root.right, nodes)
        nodes.append(root.val)

# Alternative solution using a stack
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root is None:
            return []
        
        stack = [root]
        nodes = []
        visited = set()
        
        while stack:
            node = stack[-1]
            
            if node.left and node.left not in visited:
                stack.append(node.left)
            elif node.right and node.right not in visited:
                stack.append(node.right)
            else:
                stack.pop()
                nodes.append(node.val)
                visited.add(node)
        
        return nodes
        