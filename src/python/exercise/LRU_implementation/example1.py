from collections import OrderedDict

od = OrderedDict()
od['a'] = 1
od['b'] = 2
od['c'] = 3

print("Original OrderedDict:", od['b'])

# Remove the least recently added (first) element
od.popitem(last=False)
print("After popitem(last=False):", od)