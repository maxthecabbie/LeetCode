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

import queue

class Solution:
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        vis = {0}
        q = queue.Queue()
        
        for key in rooms[0]:
            vis.add(key)
            q.put(key)
        
        while not q.empty():
            room = q.get()
            
            for key in rooms[room]:
                if key not in vis:
                    vis.add(key)
                    q.put(key)
        
        return len(rooms) == len(vis)

"""
947. Most Stones Removed with Same Row or Column

On a 2D plane, we place stones at some integer coordinate points.  Each coordinate point may have at most one
stone.

Now, a move consists of removing a stone that shares a column or row with another stone on the grid.

What is the largest possible number of moves we can make?

Example 1:
Input: stones = [[0,0],[0,1],[1,0],[1,2],[2,1],[2,2]]
Output: 5

Example 2:
Input: stones = [[0,0],[0,2],[1,1],[2,0],[2,2]]
Output: 3

Example 3:
Input: stones = [[0,0]]
Output: 0
"""

class Solution(object):
    def removeStones(self, stones):
        """
        :type stones: List[List[int]]
        :rtype: int
        """
        moves = 0
        graph = collections.defaultdict(set)
        vis = set()
        
        for i in range(len(stones)):
            for j in range(len(stones)):
                x, y = stones[i], stones[j]
                
                if i != j and (x[0] == y[0] or x[1] == y[1]):
                    graph[i].add(j)
                    graph[j].add(i)
        
        for node in graph:
            if node not in vis:
                moves += self.dfs(node, graph, vis) - 1
        
        return moves
    
    def dfs(self, root, graph, vis):
        num_nodes = 1
        vis.add(root)
        
        for node in graph[root]:
            if node not in vis:
                num_nodes += self.dfs(node, graph, vis)
        
        return num_nodes

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

class Solution(object):
    def countComponents(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: int
        """
        count = 0
        
        graph = {node: set() for node in range(n)}
        for e in edges:
            graph[e[0]].add(e[1])
            graph[e[1]].add(e[0])
            
        vis = set()
        for node in range(n):
            if node not in vis:
                self.dfs(node, graph, vis)
                count += 1
        
        return count
    
    def dfs(self, root, graph, vis):
        vis.add(root)
        
        for node in graph[root]:
            if node not in vis:
                self.dfs(node, graph, vis)

"""
529. Minesweeper

Let's play the minesweeper game (Wikipedia, online game)!

You are given a 2D char matrix representing the game board. 'M' represents an unrevealed mine, 'E' represents
an unrevealed empty square, 'B' represents a revealed blank square that has no adjacent (above, below, left,
right, and all 4 diagonals) mines, digit ('1' to '8') represents how many mines are adjacent to this revealed
square, and finally 'X' represents a revealed mine.

Now given the next click position (row and column indices) among all the unrevealed squares ('M' or 'E'),
return the board after revealing this position according to the following rules:

1. If a mine ('M') is revealed, then the game is over - change it to 'X'.
2. If an empty square ('E') with no adjacent mines is revealed, then change it to revealed blank ('B') and all
of its adjacent unrevealed squares should be revealed recursively.
3. If an empty square ('E') with at least one adjacent mine is revealed, then change it to a digit ('1' to
'8') representing the number of adjacent mines.
4. Return the board when no more squares will be revealed.
 
Example 1:
Input: 
[['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'M', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E']]

Click : [3,0]

Output: 
[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'M', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

Explanation:
https://leetcode.com/problems/minesweeper/

Example 2:
Input: 
[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'M', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

Click : [1,2]

Output: 
[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'X', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

Explanation:
https://leetcode.com/problems/minesweeper/
 
Note:
1. The range of the input matrix's height and width is [1,50].
2. The click position will only be an unrevealed square ('M' or 'E'), which also means the input board
contains at least one clickable square.
3. The input board won't be a stage when game is over (some mines have been revealed).
4. For simplicity, not mentioned rules should be ignored in this problem. For example, you don't need to
reveal all the unrevealed mines when the game is over, consider any cases that you will win the game or flag
any squares.
"""

class Solution(object):
    def updateBoard(self, board, click):
        """
        :type board: List[List[str]]
        :type click: List[int]
        :rtype: List[List[str]]
        """        
        if board[click[0]][click[1]] == "M":
            board[click[0]][click[1]] = "X"
            return board
        
        queue = collections.deque([(click[0], click[1])])
        while queue:
            i, j = queue.popleft()
            
            if 0 <= i < len(board) and 0 <= j < len(board[0]) and\
            board[i][j] == "E":
                
                num_mines = self.get_mines(i, j, board)
                if num_mines > 0:
                    board[i][j] = str(num_mines)
                    continue
                
                board[i][j] = "B"
                for r, c in ((i-1, j-1), (i-1, j), (i-1, j+1),
                             (i, j-1), (i, j+1),
                             (i+1, j-1), (i+1, j), (i+1, j+1)):
                    queue.append((r, c))
        
        return board
    
    def get_mines(self, i, j, board):
        num_mines = 0
        
        for r, c in ((i-1, j-1), (i-1, j), (i-1, j+1),
                     (i, j-1), (i, j+1),
                     (i+1, j-1), (i+1, j), (i+1, j+1)):
            
            if 0 <= r < len(board) and 0 <= c < len(board[0]) and\
            board[r][c] == "M":
                num_mines += 1
        
        return num_mines

"""
756. Pyramid Transition Matrix

We are stacking blocks to form a pyramid. Each block has a color which is a one letter string, like `'Z'`.

For every block of color `C` we place not in the bottom row, we are placing it on top of a left block of 
color `A` and right block of color `B`. We are allowed to place the block there only if `(A, B, C)` is an 
allowed triple.

We start with a bottom row of bottom, represented as a single string. We also start with a list of allowed 
triples allowed. Each allowed triple is represented as a string of length 3.

Return true if we can build the pyramid all the way to the top, otherwise false.

Example 1:
Input: bottom = "XYZ", allowed = ["XYD", "YZE", "DEA", "FFF"]
Output: true
Explanation:
We can stack the pyramid like this:
    A
   / \
  D   E
 / \ / \
X   Y   Z

This works because ('X', 'Y', 'D'), ('Y', 'Z', 'E'), and ('D', 'E', 'A') are allowed triples.

Example 2:
Input: bottom = "XXYX", allowed = ["XXX", "XXY", "XYX", "XYY", "YXZ"]
Output: false
Explanation:
We can't stack the pyramid to the top.
Note that there could be allowed triples (A, B, C) and (A, B, D) with C != D.

Note:
bottom will be a string with length in range [2, 8].
allowed will have length in range [0, 200].
Letters in all strings will be chosen from the set {'A', 'B', 'C', 'D', 'E', 'F', 'G'}.
"""

class Solution:
    def pyramidTransition(self, bottom, allowed):
        """
        :type bottom: str
        :type allowed: List[str]
        :rtype: bool
        """
        amap = {}
        for a in allowed:
            amap[a[:2]] = amap.get(a[:2], []).append(a[2])
        return self.dfs(bottom, amap)
        
    def dfs(self, bot, amap):
        if len(bot) == 2:
            return bot in amap
        
        options = []
        
        for i in range(1, len(layer)):
            if bot[i-1:i+1] in amap:
                options.append(amap[bot[i-1:i+1]])
            else:
                return False
        
        for opt in options:
            if self.dfs(layer, amap):
                return True
        return False

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

import queue

class Solution:
    def wallsAndGates(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: void Do not return anything, modify rooms in-place instead.
        """
        q = queue.Queue()
        
        for i in range(len(rooms)):
            for j in range(len(rooms[0])):
                if rooms[i][j] == 0:
                    q.put((i, j))
        
        while not q.empty():
            cell = q.get()
            i, j = cell[0], cell[1]
            
            for r, c in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                if 0 <= r < len(rooms) and 0 <= c < len(rooms[0]) \
                and rooms[r][c] > rooms[i][j] + 1:
                    rooms[r][c] = rooms[i][j] + 1
                    q.put((r, c))
                    
"""
490. The Maze

There is a ball in a maze with empty spaces and walls. The ball can go through empty spaces by rolling up, 
down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose 
the next direction.

Given the ball's start position, the destination and the maze, determine whether the ball could stop at the 
destination.

The maze is represented by a binary 2D array. 1 means the wall and 0 means the empty space. You may assume 
that the borders of the maze are all walls. The start and destination coordinates are represented by row and 
column indexes.

Example 1
Input 1: a maze represented by a 2D array

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

Input 2: start coordinate (rowStart, colStart) = (0, 4)
Input 3: destination coordinate (rowDest, colDest) = (4, 4)

Output: true
Explanation: One possible way is : left -> down -> left -> down -> right -> down -> right.

Example 2
Input 1: a maze represented by a 2D array

0 0 1 0 0
0 0 0 0 0
0 0 0 1 0
1 1 0 1 1
0 0 0 0 0

Input 2: start coordinate (rowStart, colStart) = (0, 4)
Input 3: destination coordinate (rowDest, colDest) = (3, 2)

Output: false
Explanation: There is no way for the ball to stop at the destination.

Note:
There is only one ball and one destination in the maze.
Both the ball and the destination exist on an empty space, and they will not be at the same position 
initially.
The given maze does not contain border (like the red rectangle in the example pictures), but you could 
assume the border of the maze are all walls.
The maze contains at least 2 empty spaces, and both the width and height of the maze won't exceed 100.

https://leetcode.com/problems/the-maze/description/
"""

import queue

class Solution:
    def hasPath(self, maze, start, destination):
        """
        :type maze: List[List[int]]
        :type start: List[int]
        :type destination: List[int]
        :rtype: bool
        """
        q = queue.Queue()
        q.put(start)
        vis = set()
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while not q.empty():
            cell = q.get()
            
            for i in range(len(dirs)):
                coord = self.traverse(cell, dirs[i], maze, vis, q)
                if coord == destination:
                    return True
        return False
    
    def traverse(self, coord, d, mz, vis, q):
        i, j = coord[0], coord[1]

        while 0 <= i < len(mz) and 0 <= j < len(mz[0]) and mz[i][j] == 0:
            i += d[0]
            j += d[1]
        
        i, j = i-d[0], j-d[1]
        key = str(i) + ":" + str(j)
        
        if key not in vis:
            vis.add(key)
            q.put((i, j))
        return [i, j]

"""
399. Evaluate Division

Equations are given in the format A / B = k, where A and B are variables represented as strings, and k is a 
real number (floating point number). Given some queries, return the answers. If the answer does not exist, 
return -1.0.

Example:
Given a / b = 2.0, b / c = 3.0. 
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? . 
return [6.0, 0.5, -1.0, 1.0, -1.0 ].

The input is: vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> 
queries , where equations.size() == values.size(), and the values are positive. This represents the 
equations. Return vector<double>.

According to the example above:
equations = [ ["a", "b"], ["b", "c"] ],
values = [2.0, 3.0],
queries = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]. 
The input is always valid. You may assume that evaluating the queries will result in no division by zero and 
there is no contradiction.
"""

class Solution:
    def calcEquation(self, equations, values, queries):
        """
        :type equations: List[List[str]]
        :type values: List[float]
        :type queries: List[List[str]]
        :rtype: List[float]
        """
        g = {}
        for (t, b), v in zip(equations, values):
            g.setdefault(t, []).append((b, v))
            g.setdefault(b, []).append((t, 1/v))
        return [self.dfs(q[0], q[1], 1.0, g, set()) for q in queries]

    def dfs(self, start, end, res, g, visited):
        if start not in g or end not in g:
            print(start, end)
            return -1.0

        if start == end:
            return res

        visited.add(start)
        for node in g[start]:
            if node[0] not in visited:
                t = self.dfs(node[0], end, res * node[1], g, visited)
                if t != -1.0:
                    return t
        return -1.0

"""
863. All Nodes Distance K in Binary Tree

We are given a binary tree (with root node root), a target node, and an integer value K.

Return a list of the values of all nodes that have a distance K from the target node.  The answer can be
returned in any order.

Example 1:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2
Output: [7,4,1]

Explanation: 
The nodes that are a distance 2 from the target node (with value 5)
have values 7, 4, and 1.

https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/

Note that the inputs "root" and "target" are actually TreeNodes.
The descriptions of the inputs above are just serializations of these objects.

Note:
1. The given tree is non-empty.
2. Each node in the tree has unique values 0 <= node.val <= 500.
3. The target node is a node in the tree.
4. 0 <= K <= 1000.
"""

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def distanceK(self, root, target, K):
        """
        :type root: TreeNode
        :type target: TreeNode
        :type K: int
        :rtype: List[int]
        """
        res = []
        
        graph = collections.defaultdict(set)
        self.build_graph(root, graph)
        
        queue = collections.deque([(target.val, 0)])
        vis = set([target.val])
        
        while queue:
            curr_node, moves = queue.popleft()
            
            if moves == K:
                res.append(curr_node)
            
            for node in graph[curr_node]:
                if node not in vis and moves < K:
                    queue.append((node, moves + 1))
                    vis.add(node)
        
        return res
    
    def build_graph(self, root, graph): 
        left, right = root.left, root.right
        
        if left:
            graph[root.val].add(left.val)
            graph[left.val].add(root.val)
            self.build_graph(left, graph)
            
        if right:
            graph[root.val].add(right.val)
            graph[right.val].add(root.val)
            self.build_graph(right, graph)

"""
785. Is Graph Bipartite?

Given an undirected graph, return true if and only if it is bipartite.

Recall that a graph is bipartite if we can split it's set of nodes into two independent subsets A and B such
that every edge in the graph has one node in A and another node in B.

The graph is given in the following form: graph[i] is a list of indexes j for which the edge between nodes i
and j exists.  Each node is an integer between 0 and graph.length - 1.  There are no self edges or parallel
edges: graph[i] does not contain i, and it doesn't contain any element twice.

Example 1:
Input: [[1,3], [0,2], [1,3], [0,2]]
Output: true

Explanation: 
The graph looks like this:
0----1
|    |
|    |
3----2
We can divide the vertices into two groups: {0, 2} and {1, 3}.

Example 2:
Input: [[1,2,3], [0,2], [0,1,3], [0,2]]
Output: false

Explanation: 
The graph looks like this:
0----1
| \  |
|  \ |
3----2
We cannot find a way to divide the set of nodes into two independent subsets.

Note:
1. graph will have length in range [1, 100].
2. graph[i] will contain integers in range [0, graph.length - 1].
3. graph[i] will not contain i or duplicate values.
4. The graph is undirected: if any element j is in graph[i], then i will be in graph[j].
"""

class Solution(object):
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """
        color = {}
        
        for node in range(len(graph)):
            if node not in color:
                color[node] = 1
                if not self.dfs(node, graph, color):
                    return False
        
        return True
    
    def dfs(self, root, graph, color):
        for node in graph[root]:
            if node in color:
                if color[node] == color[root]:
                    return False
            else:
                color[node] = 2 if color[root] == 1 else 1
                if not self.dfs(node, graph, color):
                    return False
        
        return True

"""
743. Network Delay Time

There are N network nodes, labelled 1 to N.

Given times, a list of travel times as directed edges times[i] = (u, v, w), where u is the source node, v is
the target node, and w is the time it takes for a signal to travel from source to target.

Now, we send a signal from a certain node K. How long will it take for all nodes to receive the signal? If it
is impossible, return -1.

Note:
1. N will be in the range [1, 100].
2. K will be in the range [1, N].
3. The length of times will be in the range [1, 6000].
4. All edges times[i] = (u, v, w) will have 1 <= u, v <= N and 0 <= w <= 100.
"""

class Solution(object):
    def networkDelayTime(self, times, N, K):
        """
        :type times: List[List[int]]
        :type N: int
        :type K: int
        :rtype: int
        """
        graph = collections.defaultdict(list)
        for u, v, t in times:
            graph[u].append((v, t))
        
        heap = [(0, K)]
        vis = set()
        
        while heap:
            curr_time, u = heapq.heappop(heap)
            
            vis.add(u)
            if len(vis) == N:
                return curr_time
            
            for v, time in graph[u]:
                if v not in vis:
                    heapq.heappush(heap, (curr_time + time, v))
        
        return -1
        

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
207. Course Schedule

There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is
expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all
courses?

Example 1:
Input: 2, [[1,0]] 
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.

Example 2:
Input: 2, [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.

Note:
The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about
how a graph is represented.

You may assume that there are no duplicate edges in the input prerequisites.
"""

class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        graph = {n : set() for n in range(numCourses)}
        inc = {n : 0 for n in range(numCourses)}
        
        for p in prerequisites:
            graph[p[1]].add(p[0])
            inc[p[0]] += 1
        
        free_nodes = [node for node in inc if inc[node] == 0]
        vis = 0
        
        while free_nodes:
            free = free_nodes.pop()
            
            for node in graph[free]:
                inc[node] -= 1
                if inc[node] == 0:
                    free_nodes.append(node)
            
            vis += 1
        
        return vis == numCourses

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

import queue

class Solution:
    def pacificAtlantic(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        if len(matrix) <= 0:
            return []
        
        pac = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        atl = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
        pac_q = queue.Queue()
        atl_q = queue.Queue()
        
        for i in range(len(matrix)):
            pac_q.put((i, 0))
            atl_q.put((i, len(matrix[0]) - 1))
        
        for j in range(len(matrix[0])):
            pac_q.put((0, j))
            atl_q.put((len(matrix) - 1, j))
        
        self.bfs(pac_q, matrix, pac)
        self.bfs(atl_q, matrix, atl)
        
        res = []
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if pac[i][j] and atl[i][j]:
                    res.append([i, j])
        return res
    
    def bfs(self, q, matrix, ocean):
        while not q.empty():
            cell = q.get()
            i, j = cell[0], cell[1]
            ocean[i][j] = True
            
            for r, c in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                if 0 <= r < len(matrix) and 0 <= c < len(matrix[0]) \
                and matrix[r][c] >= matrix[i][j] and not ocean[r][c]:
                    q.put((r, c))

"""
787. Cheapest Flights Within K Stops

There are n cities connected by m flights. Each fight starts from city u and arrives at v with a price w.

Now given all the cities and flights, together with starting city src and the destination dst, your task is
to find the cheapest price from src to dst with up to k stops. If there is no such route, output -1.

Example 1:
Input: 
n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
src = 0, dst = 2, k = 1
Output: 200

Explanation: 
The graph looks like this:
https://leetcode.com/problems/cheapest-flights-within-k-stops/

The cheapest price from city 0 to city 2 with at most 1 stop costs 200, as marked red in the picture.

Example 2:
Input: 
n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
src = 0, dst = 2, k = 0
Output: 500

Explanation: 
The graph looks like this:
https://leetcode.com/problems/cheapest-flights-within-k-stops/

The cheapest price from city 0 to city 2 with at most 0 stop costs 500, as marked blue in the picture.

Note:
1. The number of nodes n will be in range [1, 100], with nodes labeled from 0 to n - 1.
2. The size of flights will be in range [0, n * (n - 1) / 2].
3. The format of each flight will be (src, dst, price).
4. The price of each flight will be in the range [1, 10000].
5. k is in the range of [0, n - 1].
6. There will not be any duplicated flights or self cycles.
"""

class Solution(object):
    def findCheapestPrice(self, n, flights, src, dst, K):
        """
        :type n: int
        :type flights: List[List[int]]
        :type src: int
        :type dst: int
        :type K: int
        :rtype: int
        """
        min_cost = sys.maxsize
        
        graph = collections.defaultdict(list)
        for u, v, cost in flights:
            graph[u].append((v, cost))
        queue = collections.deque([(src, 0, 0)])
        
        while queue:
            node, stops, cost = queue.popleft()
            if node == dst:
                min_cost = min(min_cost, cost)
            
            if stops <= K:
                for child, price in graph[node]:
                    if cost + price < min_cost:
                        queue.append((child, stops + 1, cost + price))
        
        return min_cost if min_cost != sys.maxsize else -1

"""
210. Course Schedule II

There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

Example 1:

Input: 2, [[1,0]] 
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished   
             course 0. So the correct course order is [0,1] .
Example 2:

Input: 4, [[1,0],[2,0],[3,1],[3,2]]
Output: [0,1,2,3] or [0,2,1,3]
Explanation: There are a total of 4 courses to take. To take course 3 you should have finished both     
             courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. 
             So one correct course order is [0,1,2,3]. Another correct ordering is [0,2,1,3] .
Note:

The input prerequisites is a graph represented by a list of edges, not adjacency matrices. Read more about how a graph is represented.
You may assume that there are no duplicate edges in the input prerequisites.
"""

class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        res = []
        graph = {n : set() for n in range(numCourses)}
        inc = {n : 0 for n in range(numCourses)}
        
        for p in prerequisites:
            graph[p[1]].add(p[0])
            inc[p[0]] += 1
        
        free_nodes = [node for node in inc if inc[node] == 0]
        
        while free_nodes:
            free = free_nodes.pop()
            res.append(free)
            
            for node in graph[free]:
                inc[node] -= 1
                if inc[node] == 0:
                    free_nodes.append(node)
        
        return res if len(res) == numCourses else []
   
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

import queue

class Solution:
    def updateMatrix(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[List[int]]
        """
        q = queue.Queue()
        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    q.put((i, j))
                else:
                    matrix[i][j] = sys.maxsize
        
        while not q.empty():
            cell = q.get()
            i, j = cell[0], cell[1]
            
            for r, c in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                if 0 <= r < len(matrix) and 0 <= c < len(matrix[0]) \
                and matrix[i][j] + 1 < matrix[r][c]:
                    matrix[r][c] = matrix[i][j] + 1
                    q.put((r, c))
        
        return matrix

"""
909. Snakes and Ladders

On an N x N board, the numbers from 1 to N*N are written boustrophedonically starting from the bottom left of
the board, and alternating direction each row.  For example, for a 6 x 6 board, the numbers are written as
follows:

https://leetcode.com/problems/snakes-and-ladders/

You start on square 1 of the board (which is always in the last row and first column).  Each move, starting
from square x, consists of the following:

You choose a destination square S with number x+1, x+2, x+3, x+4, x+5, or x+6, provided this number is <=
N*N.
(This choice simulates the result of a standard 6-sided die roll: ie., there are always at most 6
destinations, regardless of the size of the board.)
If S has a snake or ladder, you move to the destination of that snake or ladder.  Otherwise, you move to S.
A board square on row r and column c has a "snake or ladder" if board[r][c] != -1.  The destination of that
snake or ladder is board[r][c].

Note that you only take a snake or ladder at most once per move: if the destination to a snake or ladder is
the start of another snake or ladder, you do not continue moving.  (For example, if the board is `[[4,-1],
[-1,3]]`, and on the first move your destination square is `2`, then you finish your first move at `3`,
because you do not continue moving to `4`.)

Return the least number of moves required to reach square N*N.  If it is not possible, return -1.

Example 1:
Input: [
[-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1],
[-1,35,-1,-1,13,-1],
[-1,-1,-1,-1,-1,-1],
[-1,15,-1,-1,-1,-1]]
Output: 4

Explanation: 
At the beginning, you start at square 1 [at row 5, column 0].
You decide to move to square 2, and must take the ladder to square 15.
You then decide to move to square 17 (row 3, column 5), and must take the snake to square 13.
You then decide to move to square 14, and must take the ladder to square 35.
You then decide to move to square 36, ending the game.
It can be shown that you need at least 4 moves to reach the N*N-th square, so the answer is 4.

Note:
1. 2 <= board.length = board[0].length <= 20
2. board[i][j] is between 1 and N*N or is equal to -1.
3. The board square with number 1 has no snake or ladder.
4. The board square with number N*N has no snake or ladder.
"""

class Solution(object):
    def snakesAndLadders(self, board):
        """
        :type board: List[List[int]]
        :rtype: int
        """
        queue = collections.deque([(1, 0)])
        vis = set([1])
        n = len(board)
        
        while queue:
            cell, moves = queue.popleft()
            
            if cell == n*n:
                return moves
            
            for d in range(1, 7):
                next_cell = cell + d
                
                if next_cell <= n*n:
                    i, j = self.get_coord(next_cell, board)
                    if board[i][j] != -1:
                        next_cell = board[i][j]
                    if next_cell not in vis:
                        vis.add(next_cell)
                        queue.append((next_cell, moves + 1))
        
        return -1
    
    def get_coord(self, cell, board):
        n = len(board)
        idx = cell - 1
        last = n-1
        
        i = last - (idx // n)
        go_right = (last - i) % 2 == 0
        j = idx % n if go_right else last - (idx % n)
        
        return i, j


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

"""
127. Word Ladder

Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest
transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time.
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.

Note:
Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.

Example 1:
Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output: 5

Explanation: As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
return its length 5.

Example 2:
Input:
beginWord = "hit"
endWord = "cog"
wordList = ["hot","dot","dog","lot","log"]

Output: 0

Explanation: The endWord "cog" is not in wordList, therefore no possible transformation.
"""

import Queue

class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        trans_map = collections.defaultdict(list)
        
        for word in wordList:
            for i in range(len(word)):
                trans_map[word[:i] + "*" + word[i+1:]].append(word)
        
        queue = Queue.Queue()
        queue.put((beginWord, 1))
        vis = set([beginWord])
        
        while not queue.empty():
            word, curr_len = queue.get()
            
            if word == endWord:
                return curr_len
            
            for i in range(len(word)):
                for trans in trans_map[word[:i] + "*" + word[i+1:]]:
                    if trans not in vis:
                        queue.put((trans, curr_len + 1))
                        vis.add(trans)
        
        return 0

"""
130. Surrounded Regions

Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

Example:
X X X X
X O O X
X X O X
X O X X

After running your function, the board should be:

X X X X
X X X X
X X X X
X O X X

Explanation:
Surrounded regions shouldn’t be on the border, which means that any 'O' on the border of the board are not
flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be
flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.
"""
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
        """
        if not board:
            return
        
        for i in (0, len(board) - 1):
            for j in range(len(board[0])):
                if board[i][j] == "O":
                    self.dfs(i, j, board)

        for j in (0, len(board[0]) - 1):
            for i in range(len(board)):
                if board[i][j] == "O":
                    self.dfs(i, j, board)

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == "O":
                    board[i][j] = "X"
                elif board[i][j] == "#":
                    board[i][j] = "O"

    def dfs(self, i, j, board):
        if (0 <= i < len(board) and 0 <= j < len(board[0]) and
            board[i][j] == "O"):

            board[i][j] = "#"

            for r, c in ((i, j-1), (i-1, j), (i, j+1), (i+1, j)):
                self.dfs(r, c, board)