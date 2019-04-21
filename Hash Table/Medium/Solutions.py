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
        
