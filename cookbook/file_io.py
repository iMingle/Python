"""Files and I/O.

"""

#1 读写文本数据
with open("data/somefile.txt", "rt") as f:
    data = f.read()
    print(data)
with open("data/somefile.txt", "rt", encoding="utf-8") as f:
    for line in f:
        print(line)
# 写入文件
with open("data/write.txt", "wt", encoding="utf-8") as f:
    f.write("content1")
    f.write("content2\u0664")
with open("data/write.txt", "wt", errors="ignore") as f:
    print("content1", file=f)
    print("content2\u0664", file=f)

#2 将输出重定向到文件中
with open("data/write.txt", "wt", errors="ignore") as f:
    print("content1", file=f)
    print("content2\u0664", file=f)

#3 以不同的分隔符或行结尾符完成打印
print("ACME", 50, 91.5)
print("ACME", 50, 91.5, sep=",")
print("ACME", 50, 91.5, sep=",", end="!!\n")
for i in range(5):
    print(i)
for i in range(5):
    print(i, end=" ")

row = ["ACME", 50, 91.5]
print(",".join(str(x) for x in row))
print(*row, sep=",")

#4 读写二进制数据
with open("data/data.bin", "rb") as f:
    data = f.read()
    print(data)
with open("data/data.bin", "wb") as f:
    f.write(b"Hello World")

t = "Hello World"
print(t[0])
for c in t:
    print(c)
b = b"Hello World"
print(b[0])
for c in b:
    print(c)

with open("data/data.bin", "rb") as f:
    data = f.read(16)
    text = data.decode("utf-8")
    print(text)
with open("data/data.bin", "wb") as f:
    text = "Hello World"
    f.write(text.encode("utf-8"))

# 写入数组
import array
nums = array.array("i", [1, 2, 3, 4])
with open("data/data.bin", "wb") as f:
    f.write(nums)
    print([num for num in nums])
a = array.array("i", [0, 0, 0, 0, 0, 0, 0, 0])
with open("data/data.bin", "rb") as f:
    f.readinto(a)
print(a)

#5 对已不存在的文件执行写入操作
try:
    with open("data/somefile.txt", "xt") as f:
        f.write("Hello\n")
except FileExistsError:
    print("file already exists!")
import os
if not os.path.exists("data/somefile.txt"):
    with open("data/somefile.txt", "wt") as f:
        f.write("Hello\n")
else:
    print("file already exists!")

#6 在字符串上执行I/O操作
import io
s = io.StringIO()
s.write("Hello World\n")
print("this is a test", file=s)
print(s.getvalue())
s = io.StringIO("Hello\nWorld\n")
print(s.read(4))
print(s.read())

s = io.BytesIO()
s.write(b"binary data")
print(s.getvalue())

#7 读写压缩的数据文件
import gzip
with gzip.open("data/access-log.gz", "rt") as f:
    text = f.read()
    print(text)

import bz2
with bz2.open("data/access-log.bz2", "rt") as f:
    text = f.read()
    print(text)

with gzip.open("data/write.gz", "wt", compresslevel=5) as f:
    f.write("Hello World")

# 对已经打开的二进制模式进行叠加操作
f = open("data/access-log.gz", "rb")
with gzip.open(f, "rt") as g:
    text = g.read()
    print(text)

#8 对固定大小的记录进行迭代
from functools import partial
RECORD_SIZE = 2
with open("data/data.bin", "rb") as f:
    records = iter(partial(f.read, RECORD_SIZE), b"")
    for r in records:
        print(r)

#9 将二进制数据读取到可变缓冲区中
# 将数据读取到可变数组中,使用文件对象的readinto()方法即可
import os.path

def read_into_buffer(filename):
    buf = bytearray(os.path.getsize(filename))
    with open(filename, "rb") as f:
        f.readinto(buf)
    return buf
with open("data/data.bin", "wb") as f:
    f.write(b"Hello World")
buf = read_into_buffer("data/data.bin")
print(buf)
print(buf[0:5])
with open("data/newdata.bin", "wb") as f:
    f.write(buf)
# 内存映像
m1 = memoryview(buf)
m2 = m1[-5:]
print(m2)
m2[:] = b"WORLD"
print(buf)

#10 对二进制文件做内存映射
import mmap

def memory_map(filename, access=mmap.ACCESS_WRITE):
    size = os.path.getsize(filename)
    fd = os.open(filename, os.O_RDWR)
    return mmap.mmap(fd, size, access=access)

# 初始化文件
size = 1000000
with open("data/bigdata.bin", "wb") as f:
    f.seek(size-1)
    f.write(b"\x00")
m = memory_map("data/bigdata.bin")
print(len(m))
print(m[0:10])
print(m[0])
m[0:11] = b"Hello World"
m.close()
with open("data/bigdata.bin", "rb") as f:
    print(f.read(11))

with memory_map("data/bigdata.bin") as m:
    print(len(m))
    print(m[0:11])
print(m.closed)

m = memory_map("data/bigdata.bin")
v = memoryview(m).cast("I")
v[0] = 7
print(m[0:4])
m[0:4] = b"\x07\x01\x00\x00"
print(v[0])

#11 处理路径名
path = "/users/data.csv"
print(os.path.basename(path))
print(os.path.dirname(path))
print(os.path.join("tmp", "data", os.path.basename(path)))
path = "~/data/data.csv"
print(os.path.expanduser(path))
print(os.path.splitext(path))

#12 检测文件是否存在
print(os.path.exists("data/somefile.txt"))
print(os.path.exists("/etc/passwd"))
print(os.path.isfile("data/somefile.txt"))
print(os.path.isdir("data/somefile.txt"))
print(os.path.islink("data/somefile.txt"))
print(os.path.realpath("data/somefile.txt"))
print(os.path.getsize("data/somefile.txt"))
print(os.path.getmtime("data/somefile.txt"))
import time
print(time.ctime(os.path.getmtime("data/somefile.txt")))

#13 获取目录内容的列表
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
names = os.listdir("data")
print(names)
# get all regular files
names = [name for name in os.listdir("data") if os.path.isfile(os.path.join("data", name))]
print(names)
dirnames = [name for name in os.listdir("data") if os.path.isdir(os.path.join("data", name))]
print(dirnames)
pyfiles = [name for name in os.listdir(".") if name.endswith(".py")]
print(pyfiles)

# 文件名的匹配
import glob
pyfiles = glob.glob("./*.py")
print(pyfiles)
from fnmatch import fnmatch
pyfiles = [name for name in os.listdir(".") if fnmatch(name, "*.py")]
print(pyfiles)

# get file sizes and mofification dates
name_sz_date = [(name, os.path.getsize(name), os.path.getmtime(name)) for name in pyfiles]
for name, size, date in name_sz_date:
    print(name, size, date)
# get file metadata
file_metadata = [(name, os.stat(name)) for name in pyfiles]
for name, meta in file_metadata:
    print(name, meta.st_size, meta.st_mtime)

#15 绕过文件名编码
import sys
print(sys.getfilesystemencoding())
# write a file using a unicode filename
with open("data/jalape\xf1o.txt", "w") as f:
    f.write("Spicy!")
print(os.listdir("data/"))
print(os.listdir(b"data/"))
with open("data/jalape\xf1o.txt", "r") as f:
    print(f.read())

#15 打印无法解码的文件名
def bad_filename(filename):
    temp = filename.encode(sys.getfilesystemencoding(), errors="ignore")
    return temp.decode("latin-1")
filename = "data/jalape\xf1o.txt"
try:
    print(filename)
except UnicodeEncodeError:
    print(bad_filename(filename))

#16 为已经打开的文件添加或修改编码方法
import urllib.request
u = urllib.request.urlopen("http://www.python.org")
f = io.TextIOWrapper(u, encoding="utf-8")
text = f.read()
print(text)
print(sys.stdout.encoding)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
print(sys.stdout.encoding)

f = open("data/data.bin", "w")
print(f) # 文本处理层,负责编码和解码Unicode
print(f.buffer) # 缓存I/O层,负责处理二进制数据
print(f.buffer.raw) # 原始文件,代表操作系统底层的文件描述符

#17 将字节数据写入文本文件
sys.stdout.buffer.write(b"Hello\n")

#18 将已有的文件描述符包装为文件对象
fd = os.open("data/write.txt", os.O_WRONLY | os.O_CREAT)
f = open(fd, "wt")
f.write("hello world\n")
f.close()

#19 创建临时文件和目录
from tempfile import TemporaryFile
with TemporaryFile("w+t") as f:
    f.write("Hello World\n")
    f.write("Testing\n")
    f.seek(0)
    data = f.read()
    print(data)
f = TemporaryFile("w+t")
f.write("Hello World\n")
f.write("Testing\n")
f.seek(0)
data = f.read()
print(data)
f.close()

from tempfile import NamedTemporaryFile
with NamedTemporaryFile("w+t") as f:
    print(f.name)

from tempfile import TemporaryDirectory
with TemporaryDirectory() as dirname:
    print(dirname)

#20 同串口进行通信
import serial

try:
    ser = serial.Serial("COM0", baudrate=9600, bytesize=8, parity="N", stopbits=1)
    ser.write(b"Hello World")
    resp = ser.readline()
    print(resp)
except Exception:
    print("serial error")

#21 序列化Python对象
import pickle
data = [1, 2, 3, 4]
f = open("data/serializing", "wb")
pickle.dump(data, f)
f.close()
s = pickle.dumps(data)
print(s)
# 读取序列化数据重建对象
f = open("data/serializing", "rb")
data = pickle.load(f)
f.close()
print(data)
data = pickle.loads(s)
print(data)

f = open("data/serializing", "wb")
pickle.dump([1, 2, 3, 4], f)
pickle.dump("hello", f)
pickle.dump({"Apple", "Pear", "Banana"}, f)
f.close()
f = open("data/serializing", "rb")
print(pickle.load(f))
print(pickle.load(f))
print(pickle.load(f))

# 某些特定类型的对象是无法进行pickle操作的,需要自定义__getstate__和__serstate__方法
import time
import threading

class Countdown:
    """Countdown Thread"""
    def __init__(self, n):
        self.n = n
        self.thr = threading.Thread(target=self.run)
        self.thr.daemon = True
        self.thr.start()

    def run(self):
        while self.n > 0:
            print("T-minus", self.n)
            self.n -= 1

    def __getstate__(self):
        return self.n

    def __setstate__(self, n):
        self.__init__(n)
c = Countdown(30)

f = open("data/thread", "wb")
pickle.dump(c, f)
f.close()
f = open("data/thread", "rb")
pickle.load(f)
