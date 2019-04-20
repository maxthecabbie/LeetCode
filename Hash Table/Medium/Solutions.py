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
        
                