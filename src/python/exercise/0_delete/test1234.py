def fibonacci_series(n):
    a,b = 0,1
    output = []

    for i in range(n):
        a,b = b,a+b
        output.append(a)
    return output

print(fibonacci_series(10))