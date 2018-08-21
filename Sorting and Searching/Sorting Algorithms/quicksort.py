def quicksort(arr):
    quicksort_h(arr, 0, len(arr)-1)

def quicksort_h(arr, lo, hi):
    if lo < hi:
        p = partition(arr, lo, hi)
        quicksort_h(arr, lo, p-1)
        quicksort_h(arr, p+1, hi)

def partition(arr, lo, hi):
    p = arr[hi]
    curr = lo

    for i in range(lo, hi):
        if arr[i] < p:
            arr[i], arr[curr] = arr[curr], arr[i]
            curr += 1
    
    arr[curr], arr[hi] = arr[hi], arr[curr]
    return curr