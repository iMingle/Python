"""Utility Scripting and System Administration.

"""

#1 通过重定向,管道或输入文件夹作为脚本的输入
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')

import fileinput

with fileinput.input('data/data.json') as f_input:
    for line in f_input:
        print(line, end='')

#2 终止程序并显示错误信息
if False:
    raise SystemError('it failed!')

#3 解析命令行选项
"""hypothetical command line tool for searching a collection of
files for one or more text patterns."""
PARSE_TEST = False
if PARSE_TEST:
    import argparse
    parser = argparse.ArgumentParser(description='Search some files')

    parser.add_argument(dest='filenames', metavar='filename', nargs='*')

    parser.add_argument('-p', '--pat', metavar='pattern', required=True,
                        dest='patterns', action='append',
                        help='text pattern to search for')

    parser.add_argument('-v', dest='verbose', action='store_true', 
                        help='verbose mode')

    parser.add_argument('-o', dest='outfile', action='store',
                        help='output file')

    parser.add_argument('--speed', dest='speed', action='store',
                        choices={'slow','fast'}, default='slow',
                        help='search speed')

    args = parser.parse_args()

    # output the collected arguments
    print(args.filenames)
    print(args.patterns)
    print(args.verbose)
    print(args.outfile)
    print(args.speed)

#4 在运行时提供密码输入提示
PASS_TEST = False
if PASS_TEST:
    import getpass

    user = getpass.getuser()
    passwd = getpass.getpass()

    print('user:', user)
    print('passwd:', passwd)

#5 获取终端大小
import shutil

sz = shutil.get_terminal_size()
print(sz)
print(sz.columns)
print(sz.lines)

#6 执行外部命令并获取输出
import subprocess

out_bytes = subprocess.check_output(['ls', '-a'])
out_text = out_bytes.decode('utf-8')
print(out_text)

try:
    out_bytes = subprocess.check_output(['ls', 'arg1', 'arg2'], stderr=subprocess.STDOUT, timeout=5)
except subprocess.CalledProcessError as e:
    out_bytes = e.output
    code = e.returncode
    out_text = out_bytes.decode('utf-8')
    print(out_text)
    print(code)

# some text to send
text = b'''
hello world
this is a test
goodbye
'''

# launch a command with pipes
p = subprocess.Popen(['wc'],
          stdout = subprocess.PIPE,
          stdin = subprocess.PIPE)

# send the data and get the output
stdout, stderr = p.communicate(text)

text = stdout.decode('utf-8')
print(text)

#7 拷贝或移动文件和目录
COPY_TEST = False
if COPY_TEST:
    shutil.copy(src, dst) # cp src dst
    # copy files, but preserve metadata
    shutil.copy2(src, dst) # cp -p src dst
    shutil.copytree(src, dst) # cp -R src dst
    shutil.move(src, dst) # mv src dst
    shutil.copy2(src, dst, follow_symlinks) # 只拷贝符号链接本身
    try:
        shutil.copytree(src, dst, symlinks=True) # 在拷贝的目录中保留符号链接
    except shutil.Error as e:
        for src, dst, msg in e.args[0]:
            print(dst, src, msg)

    def ignore_pyc_files(dirname, filenames):
        return [name in filenames if name.endswith('.pyc') else name]

    shutil.copytree(src, dst, ignore=ignore_pyc_files)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns('*~', '*.pyc'))

#8 创建和解包归档文件
ARCHIVE_TEST = False
if ARCHIVE_TEST:
    print(shutil.get_archive_formats())
    shutil.make_archive('py35', 'zip', './data')
    shutil.unpack_archive('py35.zip', './extract')

#9 通过名称来查找文件
import os

def findfile(start, name):
    for relpath, dirs, files in os.walk(start):
        if name in files:
            full_path = os.path.join(start, relpath, name)
            print(os.path.normpath(os.path.abspath(full_path)))

if __name__ == '__main__':
    findfile('.', 'data.json')

import time

def modified_within(top, seconds):
    now = time.time()
    for path, dirs, files in os.walk(top):
        for name in files:
            fullpath = os.path.join(path, name)
            if os.path.exists(fullpath):
                mtime = os.path.getmtime(fullpath)
                if mtime > (now - seconds):
                    print(fullpath)

if __name__ == '__main__':
    modified_within('.', 60*60*24*7)

#10 读取配置文件
from configparser import ConfigParser

cfg = ConfigParser()
cfg.read('data/config.ini')
print('sections:', cfg.sections())
print('installation:library', cfg.get('installation', 'library'))
print('debug:log_errors', cfg.getboolean('debug', 'log_errors'))
print('server:port', cfg.getint('server', 'port'))
print('server:nworkers', cfg.getint('server', 'nworkers'))
print('server:signature', cfg.get('server', 'signature'))

cfg.set('server', 'port', '9000')
cfg.set('debug', 'log_errors', 'False')
cfg.write(sys.stdout)

#11 给脚本添加日志记录
import logging
import logging.config

def main():
    # configure the logging system
    logging.basicConfig(
        filename='data/app.log',
        level=logging.ERROR
    )

    # variables (to make the calls that follow work)
    hostname = 'www.python.org'
    item = 'spam'
    filename = 'logging.csv'
    mode = 'r'

    # logging.config.fileConfig('data/logconfig.ini')

    logging.critical('host %s unknown', hostname)
    logging.error("couldn't find %r", item)
    logging.warning('feature is deprecated')
    logging.info('opening file %r, mode=%r', filename, mode)
    logging.debug('got here')

if __name__ == '__main__':
    main()

#12 给库添加日志记录
log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())

def func():
    log.critical('a critical error')
    log.debug('a debug message')

#13 创建一个秒表计时器
class Timer:
    def __init__(self, func=time.perf_counter):
        super(Timer, self).__init__()
        self.elapsed = 0.0
        self._func = func
        self._start = None

    def start(self):
        if self._start is not None:
            raise RuntimeError('already started')
        self._start = self._func()

    def stop(self):
        if self._start is None:
            raise RuntimeError('not started')
        end = self._func()
        self.elapsed += end - self._start
        self._start = None

    def reset(self):
        self.elapsed = 0.0

    @property
    def running(self):
        return self._start is not None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

if __name__ == '__main__':
    def countdown(n):
        while n > 0:
            n -= 1

    t = Timer()
    t.start()
    countdown(1000000)
    t.stop()
    print(t.elapsed)

    t = Timer(time.process_time)
    with t:
        countdown(1000000)
    print(t.elapsed)

#14 给内存和CPU使用量设定限制(UNIX)
import signal
# import resource

def time_exceeded(signo, frame):
    print("time's up!")
    raise SystemExit()

def set_max_runtime(seconds):
    # install the signal handler and set a resource limit
    soft, hard = resource.getrlimit(resource.RLIMIT_CPU)
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, hard))
    signal.signal(signal.SIGXCPU, time_exceeded)

def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

#15 加载Web浏览器
import webbrowser

webbrowser.open('http://www.python.org')
webbrowser.open_new('http://www.python.org')
webbrowser.open_new_tab('http://www.python.org')
try:
    c = webbrowser.get('firefox')
    c.open('http://www.python.org')
except Exception as e:
    print(e)
