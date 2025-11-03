list1 = [1,2,3,4,5,6]

list1.pop()
print(list1)

list1.append(7)
print(list1)

list2 = [8,9,10]
list3 = list1 + list2
print(list3)

list4 = [*list1 , 11]
print(list4)