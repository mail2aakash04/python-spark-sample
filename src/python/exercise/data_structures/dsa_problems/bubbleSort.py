def bubble_sort(arr) -> list :
    n = len(arr)
    for i in range(n):
        for j in range(0,n-i-1):
            if arr[j] > arr[j + 1]:
                arr[j],arr[j+1]  = arr[j+1],arr[j]
    return  arr

print(bubble_sort([5, 56, 9, 1, 4, 6]))





