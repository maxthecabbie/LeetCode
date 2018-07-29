def radix_sort(arr, base):
  if len(arr) <= 0:
    return
  max_num = max(arr)
  dig_pos = 1

  while max_num//dig_pos > 0:
      buckets = [[] for _ in range(0, base)]

      for n in arr:
          dig = n//dig_pos
          buckets[dig % base].append(n)

      i = 0
      for b in buckets:
          for n in b:
              arr[i] = n
              i += 1

      dig_pos *= base