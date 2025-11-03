from collections import Counter

def can_form_word_from_given_string1(inputStr,newStr) -> bool :

    inputStr1 = inputStr.replace(" ","").lower()
    inputStrCounter = Counter(inputStr1)


    newStrCounter = Counter(newStr)


    for ch,freq in newStrCounter.items():
        if inputStrCounter[ch] < freq:
            return False
    return True


print(can_form_word_from_given_string1("my name is aakash ","enam"))


