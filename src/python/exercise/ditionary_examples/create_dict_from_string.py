input_str = "aabbcccddeee"

# Create a dictionary from the string with each character as key and None as value
print(dict.fromkeys(input_str,None))


# Create a list from a dictionary keys
d2 = dict.fromkeys("abcdefgh")
print(d2)
finalList = [x for x in d2.keys()]
print(finalList)


