def min_freq_of_char_in_string(inputStr:str) -> tuple:
    char_freq = {}

    for ch in inputStr:
        if ch in char_freq:
            char_freq[ch] += 1
        else:
           char_freq[ch] = 1

    min_freq = min(char_freq.values())

    min_freq_ch = [ch  for ch,count in char_freq.items() if count == min_freq]
    print(min_freq_ch)
    return (min_freq_ch,min_freq)

print(min_freq_of_char_in_string("aakash"))

