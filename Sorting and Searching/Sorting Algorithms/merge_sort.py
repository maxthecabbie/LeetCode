def mergesort(arr):
    mergesort_h(arr, 0, len(arr) - 1)

def mergesort_h(arr, lo, hi):
    if lo < hi:
        mid = (lo + hi)//2
        mergesort_h(arr, lo, mid)
        mergesort_h(arr, mid+1, hi)
        merge(arr, lo, mid, hi)

def merge(arr, lo, mid, hi):
    copy = arr[:]
    p1 = lo
    p2 = mid+1
    i = lo

    while p1 <= mid and p2 <= hi:
        if copy[p1] <= copy[p2]:
            arr[i] = copy[p1]
            p1 += 1
        else:
            arr[i] = copy[p2]
            p2 += 1
        i += 1
    
    while p1 <= mid:
        arr[i] = copy[p1]
        p1 += 1
        i += 1