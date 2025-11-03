def fibonacci_problem(n):
    a, b = 0, 1
    series = []
    for i in range(n):
        series.append(a)
        a, b = b, a + b
    return series

print(fibonacci_problem(10))