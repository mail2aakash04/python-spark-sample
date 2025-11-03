from collections import Counter

def can_form_word_from_given_string(word:str,input_string:str) -> bool :

    inputChars = input_string.replace(" ","").lower()
    inputChars_count = Counter(inputChars)

    word = word.lower()
    word_count = Counter(word)

    # Check if all characters in word are available in sentence

    for ch,freq in word_count.items():
        if inputChars_count[ch] < freq:
            return False
    return  True


input = "my name is aakash"
word = "enam"

print(can_form_word_from_given_string(word,input))







