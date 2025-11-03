from collections import  Counter

def searchStringFromAnother(inputStr:str,searchStr:str) -> bool:
    out = False
    search_string_length = len(searchStr)

    for i in range(len(inputStr) - (search_string_length +1) ):
        searchStrFirstChar = searchStr[0]
        if inputStr[i:i + search_string_length] == searchStr:
            out = True
            break
        else:
            out = False
    return out

print(searchStringFromAnother("aakash","hka"))



    