def reversed_integer(x:int) :
    sign = -1 if x<0 else 1
    x = str(abs(x))
    reversed_int = x[::-1]
    return sign * int(reversed_int)

print(reversed_integer(-123))