from collections import Counter

def are_anagrams_1(str1: str, str2: str) -> bool:
    return Counter(str1) == Counter(str2)

def are_anagrams_2(str1:str,str2:str) -> bool:
    if len(str1) != len(str2):
        return False

    str1_count = {}
    for ch in str1:
        if ch in str1_count:
            str1_count[ch] +=1
        else:
            str1_count[ch] = 1

    str2_count = {}
    for ch in str2:
        if ch in str2_count:
            str2_count[ch] +=1
        else:
            str2_count[ch] = 1

    for ch,freq in str2_count.items():
        if ch not in str1_count or str1_count[ch] != freq:
            return False
    return True

print(are_anagrams_1("listen", "silent"))   # True
print(are_anagrams_1("aakash", "hasaka"))   # True
print(are_anagrams_1("hello", "world"))

print("**************************************")

print(are_anagrams_2("listen", "silent"))   # True
print(are_anagrams_2("aakash", "hasaka"))   # True
print(are_anagrams_2("hello", "world"))