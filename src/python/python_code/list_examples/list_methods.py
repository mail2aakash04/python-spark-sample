# add a single element at the end
lst = [1, 2, 3]
lst.append(4)
print(lst)

# add element at a specific index
lst = [1, 2, 3]
lst.insert(1, 10)  # insert 10 at index 1
print(lst)  # [1, 10, 2, 3]


# add multiple elements at the end
lst = [1, 2, 3]
lst.extend([4, 5, 6])
print(lst)  # [1, 2, 3, 4, 5, 6]

# Using + operator – concatenate lists
lst = [1, 2, 3]
lst = lst + [4, 5]
print(lst)  # [1, 2, 3, 4, 5]


# pop() – remove and return an element
lst = [10, 20, 30, 40]

value = lst.pop()    # removes last element
print(value)         # 40
print(lst)           # [10, 20, 30]

value = lst.pop(1)   # remove element at index 1
print(value)         # 20
print(lst)           # [10, 30]


# Using remove() – remove element by value
lst = [10, 20, 30, 20]
lst.remove(20)   # removes first occurrence of 20
print(lst)       # [10, 30, 20]

