"""
841. Keys and Rooms

There are N rooms and you start in room 0.  Each room has a distinct number in 0, 1, 2, ..., N-1, and each 
room may have some keys to access the next room. 

Formally, each room i has a list of keys rooms[i], and each key rooms[i][j] is an integer in [0, 1, ..., 
N-1] where N = rooms.length.  A key rooms[i][j] = v opens the room with number v.

Initially, all the rooms start locked (except for room 0). 

You can walk back and forth between rooms freely.

Return true if and only if you can enter every room.

Example 1:
Input: [[1],[2],[3],[]]
Output: true
Explanation:  
We start in room 0, and pick up key 1.
We then go to room 1, and pick up key 2.
We then go to room 2, and pick up key 3.
We then go to room 3.  Since we were able to go to every room, we return true.

Example 2:
Input: [[1,3],[3,0,1],[2],[0]]
Output: false
Explanation: We can't enter the room with number 2.

Note:
1 <= rooms.length <= 1000
0 <= rooms[i].length <= 1000
The number of keys in all rooms combined is at most 3000.

Solution Explanation
- Keep a set called visited
- Have a queue initialized with keys from room 0. We can add the keys to visited since they represent rooms 
we can go to
- While the queue is not empty:
    - dequeue a key
    - For the room the key opens, add the keys to the queue and the visited set if the key is not in visited
- Return if length of the visited rooms is equal to the length of rooms
"""

class Solution:
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        visited = {0}
        q = Queue()
        
        for key in rooms[0]:
            q.enqueue(key)
            visited.add(key)
        
        while not q.is_empty():
            key = q.dequeue()
            
            for new_key in rooms[key]:
                if new_key not in visited:
                    visited.add(new_key)
                    q.enqueue(new_key)
        
        return len(visited) == len(rooms)

"""
323. Number of Connected Components in an Undirected Graph

Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a 
function to find the number of connected components in an undirected graph.

Example 1:
Input: n = 5 and edges = [[0, 1], [1, 2], [3, 4]]

     0          3
     |          |
     1 --- 2    4 

Output: 2

Example 2:
Input: n = 5 and edges = [[0, 1], [1, 2], [2, 3], [3, 4]]

     0           4
     |           |
     1 --- 2 --- 3

Output:  1

Note:
You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the 
same as [1, 0] and thus will not appear together in edges.

Solution Explanation
- Initialize nodes set to be nodes 0 to n-1
- Initialize graph, which is a dictionary with node numbers as the keys and a set containing neighboring 
edges
- While the length of the nodes set is greater than 0:
    - Do DFS on any node (pop), removing each node from the nodes set as you visit it
    - Keep a count of how many times you do this, as this is the number of connected components
- Return count
"""

class Solution:
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        nodes = {i for i in range(0, n)}
        graph = {}
        
        for e in edges:
            graph.setdefault(e[0], set()).add(e[1])
            graph.setdefault(e[1], set()).add(e[0])

        count = 0
        while len(nodes) > 0:
            start = nodes.pop()
            self.dfs(start, graph, nodes)
            count += 1
        return count

    def dfs(self, n, graph, nodes):
        for neighbor in graph.get(n, set()):
            if neighbor in nodes:
                nodes.remove(neighbor)
                self.dfs(neighbor, graph, nodes)

"""
286. Walls and Gates

You are given a m x n 2D grid initialized with these three possible values.

-1 - A wall or an obstacle.
0 - A gate.
INF - Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume 
that the distance to a gate is less than 2147483647.

Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should 
be filled with INF.

Example: 
Given the 2D grid:

INF  -1  0  INF
INF INF INF  -1
INF  -1 INF  -1
  0  -1 INF INF
  
After running your function, the 2D grid should be:

  3  -1   0   1
  2   2   1  -1
  1  -1   2  -1
  0  -1   3   4

Solution Explanation
- Go through matrix and add coordinates of cells that are 0 to the queue q
- While the queue is not empty, check top, bottom, left, and right cells:
    - If cell is in bounds and the dist of the current cell + 1 is less than that cells dist:
        - We update the dist of that cell
        - Add that cell to our queue
- NOTE we do not want to scan through the matrix and do a BFS when we encounter a 0, instead we want to add 
the cell to our queue. This is to avoid repeated visiting of the same cells. In the worst case, we could 
have a bunch of 0 cells far away from a bunch of 1 cells. We'd then visit those cells multiple times to 
update their values from closer 0 cells.
"""

class Solution:
    def wallsAndGates(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: void Do not return anything, modify rooms in-place instead.
        """
        q = Queue()
        
        for i in range(len(rooms)):
            for j in range(len(rooms[0])):
                if rooms[i][j] == 0:
                    q.enqueue([i, j])
        
        while not q.is_empty():
            cell = q.dequeue()
            i, j = cell[0], cell[1]
            dist = rooms[i][j]
            
            for r, c in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                if 0 <= r < len(rooms) and 0 <= c < len(rooms[0]) \
                and rooms[r][c] > dist + 1:
                    rooms[r][c] = dist + 1
                    q.enqueue([r, c])

"""
200. Number of Islands

Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by 
water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges 
of the grid are all surrounded by water.

Example 1:
Input:
11110
11010
11000
00000

Output: 1

Example 2:
Input:
11000
11000
00100
00011

Output: 3
"""

class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        count = 0
        
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    count += 1
                    self.dfs(grid, i, j)
        return count
        
    def dfs(self, grid, i, j):
        for r, c in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) \
            and grid[r][c] == "1":
                grid[r][c] = "0"
                self.dfs(grid, r, c)

"""
417. Pacific Atlantic Water Flow

Given an m x n matrix of non-negative integers representing the height of each unit cell in a continent, the 
"Pacific ocean" touches the left and top edges of the matrix and the "Atlantic ocean" touches the right and 
bottom edges.

Water can only flow in four directions (up, down, left, or right) from a cell to another one with height 
equal or lower.

Find the list of grid coordinates where water can flow to both the Pacific and Atlantic ocean.

Note:
The order of returned grid coordinates does not matter.
Both m and n are less than 150.

Example:

Given the following 5x5 matrix:

  Pacific ~   ~   ~   ~   ~ 
       ~  1   2   2   3  (5) *
       ~  3   2   3  (4) (4) *
       ~  2   4  (5)  3   1  *
       ~ (6) (7)  1   4   5  *
       ~ (5)  1   1   2   4  *
          *   *   *   *   * Atlantic

Return:

[[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (positions with parentheses in above matrix).

Solution Explanation
- pacific and atlantic arrays keep track of which cells in the matrix can reach the pacific and atlantic 
ocean respectively
- Create two queues, one for checking which cells can reach the pacfic ocean, and one for checking which 
cells can reach the atlantic ocean
- For the pacific queue, put in cells from first row and first column since these touch the pacific
- For atlantic, put cells from last row and last column since these touch the atlantic
- Now do two BFS, one using the pacific queue and the other using the atlantic queue
    - Only enqueue the top, bottom, left, and right cells if their height is greater than or equal to 
    current cell
- Iterate through matrix, and if that cell is true in both the pacific matrix and atlantic matrix, add that 
cell to the result
"""

class Solution:
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if len(matrix) <= 0:
            return []
        
        result = []
        pacific = [[False for j in range(len(matrix[0]))] for i in range(len(matrix))]
        atlantic = [[False for j in range(len(matrix[0]))] for i in range(len(matrix))]
        pq = Queue()
        aq = Queue()
        
        for i in range(len(matrix)):
            pq.enqueue([i, 0])
            aq.enqueue([i, len(matrix[0]) - 1])
        
        for j in range(len(matrix[0])):
            pq.enqueue([0, j])
            aq.enqueue([len(matrix) - 1, j])
        
        self.bfs(pq, matrix, pacific)
        self.bfs(aq, matrix, atlantic)

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if pacific[i][j] and atlantic[i][j]:
                    result.append([i, j])
        return result
    
    def bfs(self, q, matrix, ocean):
        while not q.is_empty():
            cell = q.dequeue()
            i, j = cell[0], cell[1]
            h = matrix[i][j]
            ocean[i][j] = True
            
            for r, c in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                if 0 <= r < len(matrix) and 0 <= c < len(matrix[0]) \
                and matrix[r][c] >= h and not ocean[r][c]:
                    q.enqueue([r, c]) 
            
"""
542. 01 Matrix

Given a matrix consists of 0 and 1, find the distance of the nearest 0 for each cell.

The distance between two adjacent cells is 1.

Example 1: 
Input:
0 0 0
0 1 0
0 0 0

Output:
0 0 0
0 1 0
0 0 0

Example 2: 
Input:
0 0 0
0 1 0
1 1 1

Output:
0 0 0
0 1 0
1 2 1

Note:
The number of elements of the given matrix will not exceed 10,000.
There are at least one 0 in the given matrix.
The cells are adjacent in only four directions: up, down, left and right.
"""

class Solution:
    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        q = []
        m, n = len(matrix), len(matrix[0])
        
        result = [[0 for j in range(n)] for i in range(m)]
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    q.append([i, j])
                else:
                    matrix[i][j] = sys.maxsize
        
        for cell in q:
            i, j = cell[0], cell[1]
            for r, c in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                val = matrix[i][j] + 1
                if 0 <= r < m and 0 <= c < n and val < matrix[r][c]:
                    matrix[r][c] = val
                    q.append([r, c])
        
        return matrix

"""
332. Reconstruct Itinerary

Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], 
reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the 
itinerary must begin with JFK.

Note:
If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order 
when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than 
["JFK", "LGB"].
All airports are represented by three capital letters (IATA code).
You may assume all tickets form at least one valid itinerary.

Example 1:
Input: tickets = [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
Output: ["JFK", "MUC", "LHR", "SFO", "SJC"]

Example 2:
Input: tickets = [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"]. But it is larger in 
lexical order.
"""

class Solution:
    def findItinerary(self, tickets):
        """
        :type tickets: List[List[str]]
        :rtype: List[str]
        """
        result = []
        graph = {}
        
        for t in tickets:
            graph.setdefault(t[0], []).append(t[1])
            
        for key in graph:
            graph[key].sort(reverse = True)
            
        self.dfs("JFK", graph, result)
        return result[::-1]
    
    def dfs(self, start, graph, result):
        destinations = graph.get(start, [])
        
        while len(destinations) > 0:
            city = destinations.pop()
            self.dfs(city, graph, result)
            
        result.append(start)

"""
79. Word Search

Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those 
horizontally or vertically neighboring. The same letter cell may not be used more than once.

Example:
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
"""

class Solution:
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0] and self.dfs(i, j, board, word):
                    return True
        return False
    
    def dfs(self, i, j, board, word):
        if len(word) <= 0:
            return True
        
        if i < 0 or j < 0 or i >= len(board) or j >= len(board[0]) \
        or board[i][j] != word[0]:
            return False
        
        temp = board[i][j]
        board[i][j] = "#"

        res = self.dfs(i-1, j, board, word[1:]) or self.dfs(i+1, j, board, word[1:])\
        or self.dfs(i, j-1, board, word[1:]) or self.dfs(i, j+1, board, word[1:])
        
        board[i][j] = temp
        return res

"""
133. Clone Graph

Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.


OJ's undirected graph serialization:
Nodes are labeled uniquely.

We use # as a separator for each node, and , as a separator for node label and each neighbor of the node.
As an example, consider the serialized graph {0,1,2#1,2#2,2}.

The graph has a total of three nodes, and therefore contains three parts as separated by #.

First node is labeled as 0. Connect node 0 to both nodes 1 and 2.
Second node is labeled as 1. Connect node 1 to node 2.
Third node is labeled as 2. Connect node 2 to node 2 (itself), thus forming a self-cycle.
Visually, the graph looks like the following:

       1
      / \
     /   \
    0 --- 2
         / \
         \_/

Solution Explanation
- To clone graph use BFS
- Initially create a new graph node that has the same value of the node given to the function and add these 
as a pair to a queue
- While the queue is not empty:
    - dequeue a pair and add the label for that pair to visited
    - For the non-copy node, loop through the neighbors and create a new copy for each neighbor to add to 
    the neighbors of the copy
        - If the label of the neighbor is not in visited, add it to the queue as a pair with the neighbor 
        copy
- Return the initial copy node
"""

# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []
class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):
        if node is None:
            return None
        
        cpy = UndirectedGraphNode(node.label)
        q = Queue()
        q.enqueue([node, cpy])
        visited = set()
        
        while not q.is_empty():
            pair = q.dequeue()
            visited.add(pair[0].label)
            
            for neighbor in pair[0].neighbors:
                neighbor_cpy = UndirectedGraphNode(neighbor.label)
                pair[1].neighbors.append(neighbor_cpy)
                
                if neighbor.label not in visited:
                    q.enqueue([neighbor, neighbor_cpy])
                    
        return cpy
                