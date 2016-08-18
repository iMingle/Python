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
print(shutil.get_archive_formats())
shutil.make_archive('py35', 'zip', './data')
shutil.unpack_archive('py35.zip', './extract')

