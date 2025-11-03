from collections import Counter

def canMakeStringFromAnother(str1:str,str2:str) -> bool :
    count1 = Counter(str1)
    count2 = Counter(str2)

    print(count1)
    print(count2)

    # Check if str1 has enough characters for str2
    for ch,freq in count2.items():
        if(count1[ch]) < freq:
            print("**************",count1[ch])
            return  False
    return True

print(canMakeStringFromAnother("aakash","hzs"))