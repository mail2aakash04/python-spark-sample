
def string_reverse1(input_str:str) -> str:
    reversedStr = ""
    for ch in input_str:
        reversedStr = ch + reversedStr
    return reversedStr

def string_reverse2(input_str:str) -> str:
    input_chars = list(input_str)
    a = 0
    b = len(input_str) - 1
    while a < b:
        temp = input_chars[a]
        input_chars[a] = input_chars[b]
        input_chars[b] = temp
        a += 1
        b -= 1

    outString = "".join(input_chars)
    return outString

def string_reverse3(input_str:str) -> str:
    return input_str[::-1]

def string_reverse4(input_str:str) -> str:
    input_chars = list(input_str)
    a = 0
    b = len(input_str) - 1
    while a < b:
        input_chars[a], input_chars[b] = input_chars[b], input_chars[a]
        a += 1
        b -= 1

    outString = "".join(input_chars)
    return outString

print(string_reverse1("abcdefg"))
print(string_reverse2("abcdefg"))
print(string_reverse3("abcdefg"))
print(string_reverse4("abcdefg"))