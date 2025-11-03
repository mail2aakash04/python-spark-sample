input_str = "aabbcccddeee"

#---------------------------------------------------------
# create a dictionary using dict() function
dict_aakash = dict(name='aakash', city='bangalore', age=30)
print(dict_aakash)

#---------------------------------------------------------
# Updating element in dictionary
dict_aakash['name'] = 'prakash'

# Adding element in dictionary
dict_aakash['country'] = 'India'
print(dict_aakash)

# Updating multiple elements in dictionary
dict_aakash.update({'name': 'Aakash', 'age': 31,'surname':'Agrawal'})
print(dict_aakash)

# remove the element from dictionary using key
dict_aakash.pop('surname')
print(dict_aakash)

# remove the last added element from dictionary
dict_aakash.popitem()
print(dict_aakash)

