#######################
#                     #
#   DATA STRUCTURES   #
#                     #
#######################

# DEQUE
d = collections.deque
d.append(x)
d.appendleft(x)
d.pop()
d.popleft()
d[0]
d[-1]

# QUEUE
q = queue.Queue()
q.get()
q.put(x)
q.empty()

# HEAP
heap = []
heapq.heapify(heap)
heapq.heappush(heap, x)
heapq.heappop()
heapq.heapreplace(heap, x)

##########################
#                        #
#   BUILT-IN FUNCTIONS   #
#                        #
##########################

# ITERTOOLS PRODUCT
for x in itertools.product(it1, it2, it3):
	print(x)

list_of_lists = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
for x in itertools.product(*list_of_lists):
	print(x)

# ZIP
for x in zip([1, 2, 3], [1, 2, 3], [1, 2, 3]):
	print(x)

list_of_lists = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
for x in zip(*list_of_lists):
	print(x)

# BISECT INSORT
arr = []
bisect.insort(arr, x)

##############
#            #
#   Tricks   #
#            #
##############

# TYPE CHECKING
type(x) is int
type(x) is str
type(x) is float
type(x) is list
type(x) is dict

# INTERVAL OVERLAP
x1 < y2 and x2 < y1
x1 <= y2 and x2 <= y1
