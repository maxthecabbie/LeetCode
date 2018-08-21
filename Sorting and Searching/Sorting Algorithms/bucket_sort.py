def bucket_sort(arr):
    if len(arr) <= 0:
        return
        
    min_val, max_val = arr[0], arr[0]
    
    for n in arr:
        min_val = min(min_val, n)
        max_val = max(max_val, n)

    buckets = [[] for _ in range(max_val - min_val + 1)]
    for n in arr:
        buckets[n - min_val].append(n)

    i = 0
    for b in buckets:
        for n in b:
            arr[i] = n
            i += 1

