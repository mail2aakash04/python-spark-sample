def is_valid_brackets(input_str: str) -> bool :
    stack = []
    brackets_map = {')':'(' ,'}':'{' ,']':'['}

    for ch in input_str:
       if ch in (brackets_map.values()):
          print(brackets_map.values())
          stack.append(ch)
       elif ch in brackets_map.keys():
          print(brackets_map)
          if len(stack) ==0 or stack[-1] != brackets_map[ch]:
             return False
          stack.pop()
       else:
         continue
    return True

print(is_valid_brackets("()"))          # True
print(is_valid_brackets("({[]})"))      # True
print(is_valid_brackets("(]"))          # False
print(is_valid_brackets("([)]"))        # False
print(is_valid_brackets("{[]}"))        # True