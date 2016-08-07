"""Module

"""

import sys
from sys import argv
from sys import *
import os

print('The command line arguments are:')
for i in sys.argv:
    print(i)

print('\n\nThe PYTHONPATH is', sys.path, '\n')
print('Current directory is', os.getcwd())

# 模块的__name__
if __name__ == '__main__':
    print('This program is being run by itself')
else:
    print('I am being imported from another module')

# dir列出模块定义的标识符
print(dir(sys))
print(dir())
a = 5
print(dir())
del a
print(dir())
print(dir(print))