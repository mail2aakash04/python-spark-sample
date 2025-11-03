def palindrome_check(input_string: str) -> bool:
    cleansed_str = input_string.replace(" ","").lower()
    return cleansed_str == input_string[::-1]


print(palindrome_check("aakash"))
print(palindrome_check("paap"))

