"""IO

"""

import string
import pickle    # 持久地存储对象

def reverse(text):
    return text[::-1]

def is_palindrome(text):
    text = text.lower();
    text = text.replace(' ', '')
    for char in string.punctuation:
        text = text.replace(char, '')
    return text == reverse(text)

def main():
    something = input('Enter text:')
    if is_palindrome(something):
        print('Yes, {0} is a palindrome'.format(something))
    else:
        print('No, {0} is not a palindrome'.format(something))

if __name__ == '__main__':
    main()
else:
    print('io.py was imported')

# 读写文件
poem = """\
Programming is fun
When the work is done
if you wanna make your work also fun:
    use Python!
"""
f = open('./data/poem.txt', 'w')
f.write(poem)
f.close()

f = open('./data/poem.txt')
while True:
    line = f.readline()
    if len(line) == 0:
        break
    print(line, end = '')
f.close()

shoplistfile = './data/shoplist.data'
shoplist = ['apple','mango','carrot']
f = open(shoplistfile, 'wb')
pickle.dump(shoplist, f) #dump the object to a file
pickle.dump(shoplist, f) #dump the object to a file
f.close()

del shoplist # detroy the shoplist variable

f = open(shoplistfile, 'rb')    # Read back from the storage
storedlist = pickle.load(f) # load the object from the file
storedlist.append(pickle.load(f))
print(storedlist)
