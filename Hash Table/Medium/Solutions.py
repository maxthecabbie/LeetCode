"""
981. Time Based Key-Value Store

Create a timebased key-value store class TimeMap, that supports two operations.

1. set(string key, string value, int timestamp)
Stores the key and value, along with the given timestamp.

2. get(string key, int timestamp)
Returns a value such that set(key, value, timestamp_prev) was called previously, with timestamp_prev <= timestamp.
If there are multiple such values, it returns the one with the largest timestamp_prev.
If there are no values, it returns the empty string ("").
 
Example 1:
Input: inputs = ["TimeMap","set","get","get","set","get","get"], inputs = [[],["foo","bar",1],["foo",1],["foo",3],["foo","bar2",4],["foo",4],["foo",5]]
Output: [null,null,"bar","bar",null,"bar2","bar2"]

Explanation:   
TimeMap kv;   
kv.set("foo", "bar", 1); // store the key "foo" and value "bar" along with timestamp = 1   
kv.get("foo", 1);  // output "bar"   
kv.get("foo", 3); // output "bar" since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 ie "bar"   
kv.set("foo", "bar2", 4);   
kv.get("foo", 4); // output "bar2"   
kv.get("foo", 5); //output "bar2"   

Example 2:
Input: inputs = ["TimeMap","set","set","get","get","get","get","get"], inputs = [[],["love","high",10],["love","low",20],["love",5],["love",10],["love",15],["love",20],["love",25]]
Output: [null,null,null,"","high","high","low","low"]

Note:
1. All key/value strings are lowercase.
2. All key/value strings have length in the range [1, 100]
3. The timestamps for all TimeMap.set operations are strictly increasing.
4. 1 <= timestamp <= 10^7
5. TimeMap.set and TimeMap.get functions will be called a total of 120000 times (combined) per test case.
"""

class TimeMap(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.store = collections.defaultdict(list)
        

    def set(self, key, value, timestamp):
        """
        :type key: str
        :type value: str
        :type timestamp: int
        :rtype: None
        """
        self.store[key].append((value, timestamp))

    def get(self, key, timestamp):
        """
        :type key: str
        :type timestamp: int
        :rtype: str
        """
        if key in self.store:
            idx = self.search_times(key, timestamp)
            if idx != -1:
                return self.store[key][idx][0]
        return ""
        
    def search_times(self, key, timestamp):
        arr = self.store[key]
        lo, hi = 0, len(arr) - 1
        res = -1
        
        while lo <= hi:
            mid = (lo + hi) // 2
            
            if arr[mid][1] <= timestamp:
                res = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        return res
# Your TimeMap object will be instantiated and called as such:
# obj = TimeMap()
# obj.set(key,value,timestamp)
# param_2 = obj.get(key,timestamp)

"""
609. Find Duplicate File in System

Given a list of directory info including directory path, and all the files with contents in this directory,
you need to find out all the groups of duplicate files in the file system in terms of their paths.

A group of duplicate files consists of at least two files that have exactly the same content.

A single directory info string in the input list has the following format:

"root/d1/d2/.../dm f1.txt(f1_content) f2.txt(f2_content) ... fn.txt(fn_content)"

It means there are n files (f1.txt, f2.txt ... fn.txt with content f1_content, f2_content ... fn_content,
respectively) in directory root/d1/d2/.../dm. Note that n >= 1 and m >= 0. If m = 0, it means the directory
is just the root directory.

The output is a list of group of duplicate file paths. For each group, it contains all the file paths of the
files that have the same content. A file path is a string that has the following format:

"directory_path/file_name.txt"

Example 1:
Input:
["root/a 1.txt(abcd) 2.txt(efgh)", "root/c 3.txt(abcd)", "root/c/d 4.txt(efgh)", "root 4.txt(efgh)"]
Output:  
[["root/a/2.txt","root/c/d/4.txt","root/4.txt"],["root/a/1.txt","root/c/3.txt"]]
 

Note:
1. No order is required for the final output.
2. You may assume the directory name, file name and file content only has letters and digits, and the length of
file content is in the range of [1,50].
3. The number of files given is in the range of [1,20000].
4. You may assume no files or directories share the same name in the same directory.
5. You may assume each given directory info represents a unique directory. Directory path and file info are
separated by a single blank space.
 
Follow-up beyond contest:
1. Imagine you are given a real file system, how will you search files? DFS or BFS?
2. If the file content is very large (GB level), how will you modify your solution?
3. If you can only read the file by 1kb each time, how will you modify your solution?
4. What is the time complexity of your modified solution? What is the most time-consuming part and memory
consuming part of it? How to optimize?
5. How to make sure the duplicated files you find are not false positive?
"""

class Solution(object):
    def findDuplicate(self, paths):
        """
        :type paths: List[str]
        :rtype: List[List[str]]
        """
        res = []
        files = collections.defaultdict(list)
        
        for p in paths:
            directory = p.split(" ")
            root = directory[0]
            
            for f in directory[1:]:
                first_paren = f.find("(")
                name = f[:first_paren]
                content = f[first_paren + 1 : -1]
                files[content].append(root + "/" + name)
        
        for key in files:
            if len(files[key]) > 1:
                res.append(files[key])
        
        return res

"""
694. Number of Distinct Islands

Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected
4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

Count the number of distinct islands. An island is considered to be the same as another if and only if one
island can be translated (and not rotated or reflected) to equal the other.

Example 1:
11000
11000
00011
00011
Given the above grid map, return 1.

Example 2:
11011
10000
00001
11011
Given the above grid map, return 3.

Notice that:
11
1

and

 1
11
are considered different island shapes, because we do not consider reflection / rotation.

Note: The length of each dimension in the given grid does not exceed 50.
"""

class Solution(object):
    def numDistinctIslands(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        islands = set()
                
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == 1:
                    ser = []
                    self.dfs(i, j, grid, ser)
                    islands.add(":".join(ser))
        
        return len(islands)
        
    def dfs(self, i, j, grid, ser):
        grid[i][j] = 0
        dirs = ((0, 1), (1, 0), (0, -1), (-1, 0))
        
        ser.append("s")
        for d in range(len(dirs)):
            r, c = i + dirs[d][0], j + dirs[d][1]
            
            if 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 1:
                ser.append(str(d))
                self.dfs(r, c, grid, ser)
        ser.append("e")
            

"""
939. Minimum Area Rectangle

Given a set of points in the xy-plane, determine the minimum area of a rectangle formed from these points,
with sides parallel to the x and y axes.

If there isn't any rectangle, return 0.

Example 1:
Input: [[1,1],[1,3],[3,1],[3,3],[2,2]]
Output: 4

Example 2:
Input: [[1,1],[1,3],[3,1],[3,3],[4,1],[4,3]]
Output: 2
 
Note:
1 <= points.length <= 500
0 <= points[i][0] <= 40000
0 <= points[i][1] <= 40000
All points are distinct.
"""

class Solution(object):
    def minAreaRect(self, points):
        min_area = sys.maxsize
        
        cols = collections.defaultdict(set)
        for x, y in points:
            cols[x].add(y)
        
        for col1 in cols:
            col_list = list(cols[col1])
            
            for y_ind1 in range(len(col_list) - 1):
                for y_ind2 in range(y_ind1 + 1, len(col_list)):
                    
                    for col2 in cols:
                        if col1 == col2:
                            continue
                        
                        y1, y2 = col_list[y_ind1], col_list[y_ind2]
                        
                        if y1 in cols[col2] and y2 in cols[col2]:
                            area = abs(col1 - col2) * abs(y1 - y2)
                            min_area = min(min_area, area)
        
        return min_area if min_area != sys.maxsize else 0
        
"""
18. 4Sum

Given an array nums of n integers and an integer target, are there elements a, b, c, and d in nums such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.

Note:
The solution set must not contain duplicate quadruplets.

Example:
Given array nums = [1, 0, -1, 0, -2, 2], and target = 0.

A solution set is:
[
  [-1,  0, 0, 1],
  [-2, -1, 1, 2],
  [-2,  0, 0, 2]
]
"""

class Solution(object):
    def fourSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        res = []
        
        nums.sort()
        sums = collections.defaultdict(dict)
        
        for i in range(len(nums) - 1):
            for j in range(i+1, len(nums)):
                sum_key = nums[i] + nums[j]
                idx_key = (nums[i], nums[j])
                sums[sum_key][idx_key] = (i, j)
        
        for i in range(len(nums) - 3):
            if i > 0 and nums[i-1] == nums[i]:
                continue
                
            for j in range(i+1, len(nums) - 2):
                if j > i+1 and nums[j-1] == nums[j]:
                    continue
                
                need = target - (nums[i] + nums[j])
                
                for idx_key in sums[need]:
                    idx1, idx2 = sums[need][idx_key]
                    if idx1 > j:
                        res.append([nums[i], nums[j], nums[idx1], nums[idx2]])
                        
        return res