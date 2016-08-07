"""Data structure

"""

# 数据结构-列表
list = ['apple', 'orange', 'banana', 'carrot']
print('I have', len(list), 'items to purchase.')
print('These items are:', end = ' ')
for item in list:
    print(item, end = ' ')
print('\nI also have to buy rice.')
list.append('rice')
print('My shopping list is now', list)
print('I will sort my list now')
list.sort()
print('Sorted shopping list is', list)
print('The first item I will buy is', list[0])
olditem = list[0]
del list[0]
print('I bought the', olditem)
print('My shopping list is now', list)

# 数据结构-元组,不可变
tuple = ('python', 'elephant', 'penguin')
# tuple[3] = 'change'    # TypeError: 'tuple' object does not support item assignment
print('Number of animals in the tuple is', len(tuple))
new_tuple = ('monkey', 'camel', tuple)
print('Number of cages in the new tuple is', len(new_tuple))
print('All animals in new tuple are', new_tuple)
print('Animals brought from old tuple are', new_tuple[2])
print('Last animal brought from old tuple is', new_tuple[2][2])
print('Number of animals in the new tuple is', len(new_tuple) - 1 + len(new_tuple[2]))

singleton_tuple = (2,)
print(singleton_tuple)

# 数据结构-字典,只能使用不可变的对象（比如字符串）来作为字典的键，但是你可以把不可变或可变的对象作为字典的值
dict = {
    'Mingle' : 'jinminglei@yeah.net',
    'Larry' : 'larry@wall.org',
    'Matsumoto' : 'matz@ruby-lang.org',
    'Spammer' : 'spammer@hotmail.com'
}
print("Mingle's address is", dict['Mingle'])
# Deleting a key-value pair
del dict['Spammer']
print('\nThere are {0} contacts in the address-book\n'.format(len(dict)))
for name, address in dict.items():
    print('Contact {0} at {1}'.format(name, address))
# Adding a key-value pair
dict['Guido'] = 'guido@python.org'
if 'Guido' in dict: # OR dict.has_key('Guido')
    print("\nGuido's address is", dict['Guido'])

# 数据结构-序列,列表、元组和字符串都是序列
shoplist = ['apple', 'mango', 'carrot', 'banana']
name = 'mingle'

# Indexing or 'Subscription' operation
print('Item 0 is', shoplist[0])
print('Item 1 is', shoplist[1])
print('Item 2 is', shoplist[2])
print('Item 3 is', shoplist[3])
print('Item -1 is', shoplist[-1])
print('Item -2 is', shoplist[-2])
print('Character 0 is', name[0])

# Slicing on a list
print('Item 1 to 3 is', shoplist[1:3])
print('Item 2 to end is', shoplist[2:])
print('Item 1 to -1 is', shoplist[1:-1])
print('Item start to end is', shoplist[:])
# 切片的步长(默认步长是1)
print('Step is 1', shoplist[::1])
print('Step is 2', shoplist[::2])
print('Step is 3', shoplist[::3])
print('Step is 4', shoplist[::4])
print('Step is 5', shoplist[::5])

# Slicing on a string
print('characters 1 to 3 is', name[1:3])
print('characters 2 to end is', name[2:])
print('characters 1 to -1 is', name[1:-1])
print('characters start to end is', name[:])

# 数据结构-集合,没有顺序的简单对象的聚集
collections = set(['brazil', 'russia', 'india'])
print('india' in collections)
print('china' in collections)
collections_copy = collections.copy()
collections_copy.add('china')
print('china' in collections_copy)
collections_copy.issuperset(collections)
collections.remove('russia')
print(collections_copy & collections)

# 数据结构-引用,当你创建一个对象并给它赋一个变量的时候，这个变量仅仅引用那个对象，而不是表示这个对象本身
# 列表的赋值语句不创建拷贝。你得使用切片操作符来建立序列的拷贝
print('Simple Assignment')
shoplist = ['apple', 'mango', 'carrot', 'banana']
mylist = shoplist # mylist is just another name pointing to the same object!
del shoplist[0] # I purchased the first item, so I remove it from the list
print('shoplist is', shoplist)
print('mylist is', mylist)
# notice that both shoplist and mylist both print the same list without
# the 'apple' confirming that they point to the same object
print('Copy by making a full slice')
mylist = shoplist[:] # make a copy by doing a full slice
del mylist[0] # remove first item
print('shoplist is', shoplist)
print('mylist is', mylist)

# 数据结构-字符串
name = 'Mingle' # This is a string object

if name.startswith('Min'):
    print("Yes, the string starts with 'Min'")
if 'e' in name:
    print("Yes, it contains the string 'e'")
if name.find('war') != -1:
    print("Yes, it contains the string 'war'")
delimiter = '_*_'
mylist = ['Brazil', 'Russia', 'India', 'China']
print(delimiter.join(mylist))