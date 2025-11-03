from collections import Counter

def check_freq_of_chars_in_string (inputStr:str) -> dict:
    char_freq = {}
    for ch in inputStr:
        if ch in char_freq:
            char_freq[ch] += 1
        else:
            char_freq[ch] = 1
    return char_freq

print(check_freq_of_chars_in_string("aakash"))
print(Counter("aakash"))