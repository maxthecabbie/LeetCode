"""
369. Plus One Linked List

Given a non-negative integer represented as non-empty a singly linked list of digits, plus one to the 
integer.

You may assume the integer do not contain any leading zero, except the number 0 itself.

The digits are stored such that the most significant digit is at the head of the list.

Example:
Input:
1->2->3

Output:
1->2->4
"""

class Solution(object):
    def plusOne(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        cursor = head
        stack = []
        
        while cursor:
            stack.append(cursor)
            cursor = cursor.next
        
        while stack:
            node = stack.pop()
            if node.val < 9:
                node.val += 1
                return head
            node.val = 0
        
        new_head = ListNode(1)
        new_head.next = head
        return new_head

# Alternative O(1) space solution
"""
- To do this in place, iterate through the linked list and keep track of the last node in the list that has a
value that is not 9
- If non_nine None, that means all the nodes have a val of 9. We must add a new head with a val of 0, which 
will later get 1 added to the val
- Add 1 to the val of the non_nine node
- Iterate through the linked list starting at non_nine.next setting the val of each encountered node to 0
- Return the head
"""
class Solution(object):
    def plusOne(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        cursor = head
        non_nine = None
        
        while cursor:
            if cursor.val != 9:
                non_nine = cursor
            cursor = cursor.next
            
        if non_nine is None:
            new_head = ListNode(0)
            new_head.next = head
            head = new_head
            non_nine = head
            
        non_nine.val += 1
        cursor = non_nine.next
        
        while cursor:
            cursor.val = 0
            cursor = cursor.next
            
        return head
        
"""
817. Linked List Components

We are given head, the head node of a linked list containing unique integer values.

We are also given the list G, a subset of the values in the linked list.

Return the number of connected components in G, where two values are connected if they appear consecutively 
in the linked list.

Example 1:
Input: 
head: 0->1->2->3
G = [0, 1, 3]
Output: 2
Explanation: 
0 and 1 are connected, so [0, 1] and [3] are the two connected components.

Example 2:
Input: 
head: 0->1->2->3->4
G = [0, 3, 1, 4]
Output: 2
Explanation: 
0 and 1 are connected, 3 and 4 are connected, so [0, 1] and [3, 4] are the two connected components.

Note:
If N is the length of the linked list given by head, 1 <= N <= 10000.
The value of each node in the linked list will be in the range [0, N - 1].
1 <= G.length <= 10000.
G is a subset of all values in the linked list.

Solution Explanation:
- Create a set with all nodes from list G
- Iterate through linked list and when you encounter a node with val in gset:
    - Continue iterating while cursor is not None and the cursor.val is in gset
    - Increment count by 1
- Return count
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def numComponents(self, head, G):
        """
        :type head: ListNode
        :type G: List[int]
        :rtype: int
        """
        count = 0
        gset = set(G)
        cursor = head
        
        while cursor:
            if cursor.val in gset:
                while cursor.next and cursor.next.val in gset:
                    cursor = cursor.next
                count += 1
            cursor = cursor.next
        
        return count

"""
24. Swap Nodes in Pairs

Given a linked list, swap every two adjacent nodes and return its head.

You may not modify the values in the list's nodes, only nodes itself may be changed.
 
Example:
Given 1->2->3->4, you should return the list as 2->1->4->3.
"""
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        return self.swap_impl(head, None)
        
    def swap_impl(self, head, prev):
        if head is None:
            return
        
        if head.next:
            temp = head.next.next
            after_head = head.next
            
            after_head.next = head
            if prev:
                prev.next = after_head
            
            head.next = temp
            
            self.swap_impl(temp, head)
            
            head = after_head
            
        return head

"""
19. Remove Nth Node From End of List

Given a linked list, remove the n-th node from the end of list and return its head.

Example:
Given linked list: 1->2->3->4->5, and n = 2.

After removing the second node from the end, the linked list becomes 1->2->3->5.

Note:
Given n will always be valid.

Follow up:
Could you do this in one pass?
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """

        prev = head
        cursor = head
        size = 0
        
        while cursor:
            cursor = cursor.next
            size += 1
        
        cursor = head
        
        if n == size:
            head = head.next
            return head
        
        while n != size:
            prev = cursor
            cursor = cursor.next
            size -= 1
            
        prev.next = cursor.next
        return head

"""
328. Odd Even Linked List

Given a singly linked list, group all odd nodes together followed by the even nodes. Please note here we are 
talking about the node number and not the value in the nodes.

You should try to do it in place. The program should run in O(1) space complexity and O(nodes) time 
complexity.

Example 1:
Input: 1->2->3->4->5->NULL
Output: 1->3->5->2->4->NULL

Example 2:
Input: 2->1->3->5->6->4->7->NULL
Output: 2->3->6->7->1->5->4->NULL

Note:
The relative order inside both the even and odd groups should remain as it was in the input.
The first node is considered odd, the second node even and so on ...
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy_odd = odd = ListNode(0)
        dummy_even = even = ListNode(0)
        
        while head:
            odd.next = head
            even.next = head.next
            odd = odd.next
            even = even.next
            head = head.next.next if head.next else head.next
            
        odd.next = dummy_even.next
        return dummy_odd.next

"""
148. Sort List

Sort a linked list in O(n log n) time using constant space complexity.

Example 1:
Input: 4->2->1->3
Output: 1->2->3->4

Example 2:
Input: -1->5->3->4->0
Output: -1->0->3->4->5
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None or head.next is None:
            return head
        
        fast = head
        slow = head
        prev = None
        
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
            
        prev.next = None
        
        l = self.sortList(head)
        r = self.sortList(slow)
        return self.merge(l, r)
    
    def merge(self, l, r):
        dummy = ListNode(0)
        p = dummy
        
        while l and r:
            if l.val <= r.val:
                p.next = l
                l = l.next
            else:
                p.next = r
                r = r.next
            p = p.next
        
        if l:
            p.next = l
        
        elif r:
            p.next = r
        
        return dummy.next

"""
143. Reorder List

Given a singly linked list L: L0→L1→…→Ln-1→Ln,
reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…

You may not modify the values in the list's nodes, only nodes itself may be changed.

Example 1:
Given 1->2->3->4, reorder it to 1->4->2->3.

Example 2:
Given 1->2->3->4->5, reorder it to 1->5->2->4->3.
"""

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: None Do not return anything, modify head in-place instead.
        """
        if head is None:
            return head
        
        fast = slow = head
        stack = []
        
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        
        if fast:
            slow = slow.next
        
        while slow:
            stack.append(slow)
            slow = slow.next
        
        cursor = head
        
        while stack:
            temp = cursor.next
            stack[-1].next = temp
            cursor.next = stack.pop()
            cursor = temp
        
        cursor.next = None
        return head

"""
2. Add Two Numbers

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in 
reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked 
list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:
Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution:
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy_head = cursor = ListNode(0)
        carry = 0
        
        while l1 or l2 or carry:
            sum_nodes = carry
            carry = 0
            
            if l1:
                sum_nodes += l1.val
                l1 = l1.next
            if l2:
                sum_nodes += l2.val
                l2 = l2.next
            if sum_nodes > 9:
                carry = 1
                sum_nodes = sum_nodes % 10
            
            cursor.next = ListNode(sum_nodes)
            cursor = cursor.next
        
        return dummy_head.next

"""
138. Copy List with Random Pointer

A linked list is given such that each node contains an additional random pointer which could point to any 
node in the list or null. Return a deep copy of the list.
"""

# Definition for singly-linked list with a random pointer.
# class RandomListNode(object):
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None
class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        dummy_head = cursor = Node(-1, None, None)
        seen = {}
        
        while head:
            copy = seen[head] if head in seen else Node(head.val, None, None)
            seen[head] = copy
            
            if head.random in seen:
                copy.random = seen[head.random]
            elif head.random is not None:
                node = Node(head.random.val, None, None)
                seen[head.random] = node
                copy.random = node
            
            cursor.next = copy
            cursor = cursor.next
            head = head.next
        
        return dummy_head.next

# Alternative solution
def copyRandomList(self, head):
    d = {}
    m = n = head
    
    while m:
        d[m] = RandomListNode(m.label)
        m = m.next
        
    while n:
        d[n].next = d.get(n.next)
        d[n].random = d.get(n.random)
        n = n.next
        
    return d.get(head)

"""
708. Insert into a Cyclic Sorted List

Given a node from a cyclic linked list which is sorted in ascending order, write a function to insert a 
value into the list such that it remains a cyclic sorted list. The given node can be a reference to any 
single node in the list, and may not be necessarily the smallest value in the cyclic list.

If there are multiple suitable places for insertion, you may choose any place to insert the new value. After 
the insertion, the cyclic list should remain sorted.

If the list is empty (i.e., given node is null), you should create a new single cyclic list and return the 
reference to that single node. Otherwise, you should return the original given node.

The following example may help you understand the problem better:

https://leetcode.com/problems/insert-into-a-cyclic-sorted-list/description/
"""

"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, next):
        self.val = val
        self.next = next
"""
class Solution(object):
    def insert(self, head, insertVal):
        """
        :type head: Node
        :type insertVal: int
        :rtype: Node
        """
        if not head: 
            new_node = Node(insertVal, None)
            new_node.next = new_node
            return new_node
        
        cursor = head.next
        
        while not self.between(cursor.val, cursor.next.val, insertVal) \
        and cursor is not head:
            cursor = cursor.next
        
        node = Node(insertVal, cursor.next)
        cursor.next = node
        return head
        
    def between(self, lo, hi, t):
        if lo <= hi:
            return lo <= t <= hi
        return t >= lo or t <= hi
    
        
        