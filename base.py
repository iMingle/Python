"""Python base

"""

import pdb

print(2)
print(2.1)
print(2.1E-2)
print(4 + 5j)

print('1')
print("2")
print('''123
	345
	2354''')
print("hello,\
world")
print(R"Newlines are indicated by \n")
print(r"Newlines are indicated by \n")

# pdb.set_trace()

age = 25
name = "mingle"
print("{0} is {1} years old".format(name, age))
print('{0:.3}'.format(1/3))
print('{0:_^11}'.format('hello'))
print('{name} wrote {book}'.format(name='Mingle', book='Python'))

s = "This is a string. \
This continues the string."
print(s)

print(5 + 4)	# 9
print(5 - 4)	# 1
print(5 * 4)	# 20
print(2 ** 4)	# 16
print(5 / 4)	# 1.25
print(5 // 2)	# 2
print(5 % 4)	# 1
print(2 << 2)	# 8
print(11 >> 1)	# 5
print(5 & 3)	# 1
print(5 | 3)	# 7
print(5 ^ 3)	# 6
print(~5)		# -6
print(5 < 3)	# False
print(5 > 3)	# True
print(5 <= 3)	# False
print(5 >= 3)	# True
print(5 == 5)	# True
print(5 != 5)	# False
print(not True)	# False
print(True and False)	# False
print(True or False)	# True

length = 5
breadth = 2
area = length * breadth
print("Area is", area)

number = 23
guess = int(input('Enter an integer : '))

if guess == number:
	print('Congratulations, you guessed it.')
	print('(but you do not win any prizes!)')
elif guess < number:
	print('No, it is a little higher than that')
else:
	print('No, it is a little lower than that')
print('if Done')

running = True
while running:
	guess = int(input('Enter an integer : '))
	if guess == number:
		print('Congratulations, you guessed it.')
		running = False
	elif guess < number:
		print('No, it is a little higher than that')
	else:
		print('No, it is a little lower than that')
else:
	print("the while loop is over.")
print('while Done')

for i in range(1, 5):
	print(i)
else:
	print('The for loop is over')
print('for Done')

while True:
	s = (input('Enter something : '))
	if s == 'quit':
		break
	print('Length of the string is', len(s))
print('while break Done')

while True:
	s = input('Enter something : ')
	if s == 'quit':
		break
	if len(s) < 3:
		print("Too small")
		continue
	print('Input is of sufficient length')

def printMax(a, b):
	if a > b:
		print(a, "is maximum")
	elif a == b:
		print(a, "is equal to", b)
	else:
		print(b, "is maximum")

printMax(3, 4)

x = 50
def func(x):
	print("x is", x)
	x = 2
	print("Changed local x to", x)
func(x)
print("x is still", x)

def func():
	global x
	print("x is", x)
	x = 2
	print("Changed local x to", x)
func()
print("Value of x is", x)

def func_outer():
	x = 2
	print("x is", x)

	def func_inner():
		nonlocal x
		x = 5

	func_inner()
	print("Changed local x to", x)	# 5
func_outer()

def say(message, times = 1):
	print(message * times)
say("hello")
say("world", 5)

def func(a, b = 5, c = 10):
	print("a is", a, "and b is", b, "and c is", c)
func(3, 7)
func(25, c = 24)
func(c = 50, a = 100)

# 可变参,列表和字典
def total(initial = 5, *numbers, **keywords):
	count = initial
	for number in numbers:
		count += number
	for key in keywords:
		count += keywords[key]
	return count
print(total(10, 1, 2, 3, vegetables = 50, fruits = 100))	# 166

# keyword-only参数
def total(initial = 5, *numbers, vegetables):
	count = initial
	for number in numbers:
		count += number
	count += vegetables
	return count
print(total(10, 1, 2, 3, vegetables = 50))	# 66
# print(total(10, 1, 2, 3,))	# total() missing 1 required keyword-only argument: 'vegetables'

# 每个函数都在结尾暗含有return None语句
def nullFunc():
	pass	# 表示一个空的语句块
print(nullFunc())	# None

# DocStrings 文档字符串
def printMax(x, y):
	'''Prints the maximum of two numbers.

	The two values must be integers.'''
	x = int(x) # convert to integers, if possible
	y = int(y)

	if x > y:
		print(x, 'is maximum')
	else:
		print(y, 'is maximum')
printMax(3, 5)
print(printMax.__doc__)