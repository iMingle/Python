# Copyright (c) 2016, Mingle. All rights reserved.
# Author: Mingle
# Contact: jinminglei@yeah.net

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

