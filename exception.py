"""Exception

"""

import time

try:
	text = input('Enter something --> ')
except EOFError:
	print('Why did you do an EOF on me?')
except KeyboardInterrupt:
	print('You cancelled the operation.')
else:
	print('You entered {0}'.format(text))

class ShortInputException(Exception):
	"""A user-defined exception class"""
	def __init__(self, length, atleast):
		super(ShortInputException, self).__init__()
		self.length = length
		self.atleast = atleast

try:
	text = input('Enter something --> ')
	if len(text) < 3:
		raise ShortInputException(len(text), 3)
except EOFError:
	print('Why did you do an EOF on me?')
except ShortInputException as ex:
	print('ShortInputException The input was {0} long, excepted atleast {1}'.format(ex.length, ex.atleast))
else:
	print('No exception was raised.')

try:
	f = open('poem.txt')
	while True:
		line = f.readline()
		if len(line) == 0:
			break
		print(line, end = '')
		time.sleep(2)
except KeyboardInterrupt:
	print('!! You cancelled the reading from the file.')
finally:
	f.close()
	print('(Cleanig up: closed the file)')

# 用with open就能使得在结束的时候自动关闭文件
with open('poem.txt') as f:
	for line in f:
		print(line, end = '')