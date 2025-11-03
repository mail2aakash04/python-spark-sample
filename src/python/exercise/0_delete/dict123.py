aakash = dict(name = 'Aakash' , age = 30)

print(aakash.get('age'))    # 30

aakash['city'] = 'Delhi'
aakash['age'] = 31

print(aakash)

aakash.pop('age')

print(aakash)



if 'name' in aakash:
    print("Key exists")


for key in aakash:
    print(key, aakash[key])

print("***********")
for key, value in aakash.items():
    print(key, value)


dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged = dict1 | dict2
print(merged)