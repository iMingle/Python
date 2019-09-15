"""Testing, Debugging, and Exceptions.

"""

# 1 测试发送到stdout上的输出
from io import StringIO
from unittest import TestCase
from unittest.mock import patch


def urlprint(protocol, host, domain):
    url = '{}://{}.{}'.format(protocol, host, domain)
    print(url)


class TestURLPrint(TestCase):
    def test_url_gets_to_stdout(self):
        protocol = 'http'
        host = 'www'
        domain = 'example.com'
        expected_url = '{}://{}.{}\n'.format(protocol, host, domain)

        with patch('sys.stdout', new=StringIO()) as fake_out:
            urlprint(protocol, host, domain)
            self.assertEqual(fake_out.getvalue(), expected_url)


TEST_STDOUT = False

if __name__ == '__main__' and TEST_STDOUT:
    import unittest


    unittest.main()

# 2 在单元测试中为对象打补丁
import test_debug_exception_patch


@patch('test_debug_exception_patch.func')
def test1(x, mock_func):
    test_debug_exception_patch.func(x)
    mock_func.assert_called_with(x)


with patch('test_debug_exception_patch.func') as mock_func:
    test_debug_exception_patch.func()
    mock_func.assert_called_with()

p = patch('test_debug_exception_patch.func')
mock_func = p.start()
test_debug_exception_patch.func()
mock_func.assert_called_with()
p.stop()


@patch('test_debug_exception_patch.func')
@patch('test_debug_exception_patch.func1')
@patch('test_debug_exception_patch.func2')
def test2(mock1, mock2, mock3):
    pass


x = 42
with patch('__main__.x'):
    print(x)
print(x)

with patch('__main__.x', 'patch_value'):
    print(x)
print(x)

from unittest.mock import MagicMock


m = MagicMock(return_value=10)
print(m(1, 2, debug=True))
m.assert_called_with(1, 2, debug=True)
try:
    m.assert_called_with(1, 2)
except AssertionError as e:
    print(e)

m.upper.return_value = 'HELLO'
print(m.upper('hello'))
assert m.upper.called
m.split.return_value = ['hello', 'world']
print(m.split('hello world'))
print(m.split.assert_called_with('hello world'))
print(m['blah'])
print(m.__getitem__.called)
print(m.__getitem__.assert_called_with('blah'))

# 3 在单元测试中检测异常情况
import unittest


def parse_int(s):
    return int(s)


class TestConversion(unittest.TestCase):
    # testing that an exception gets raised
    def test_bad_int(self):
        self.assertRaises(ValueError, parse_int, "N/A")

    # testing an exception plus regex on exception message
    def test_bad_int_msg(self):
        self.assertRaisesRegex(ValueError, 'invalid literal .*', parse_int, 'N/A')


import errno


class TestIO(unittest.TestCase):
    def test_file_not_found(self):
        try:
            f = open('/file/not/found')
        except IOError as e:
            self.assertEqual(e.errno, errno.ENOENT)
        else:
            self.fail('IOError not raised')


TEST_RAISE = False

if __name__ == '__main__' and TEST_RAISE:
    unittest.main()

# 4 将测试结果作为日志记录到文件中
import sys


def main(out=sys.stderr, verbosity=2):
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    unittest.TextTestRunner(out, verbosity=verbosity).run(suite)


if __name__ == '__main__':
    with open('data/testing.out', 'w') as f:
        main(f)

# 5 跳过测试,或者预计测试结果为失败
import os
import platform


class Tests(unittest.TestCase):
    def test_0(self):
        self.assertTrue(True)

    @unittest.skip('skipped test')
    def test_1(self):
        self.fail("should have failed!")

    @unittest.skipIf(os.name == 'posix', 'Not supported on Unix')
    def test_2(self):
        pass

    @unittest.skipUnless(platform.system() == 'Darwin', 'Mac specific test')
    def test_3(self):
        self.assertTrue(True)

    @unittest.expectedFailure
    def test_4(self):
        self.assertEqual(2 + 2, 5)


TEST_SKIP = False

if __name__ == '__main__' and TEST_SKIP:
    unittest.main(verbosity=2)

# 6 处理多个异常
try:
    pass
except (URLError, ValueError):
    pass
except SocketTimeout as e:
    raise e
print(FileNotFoundError.__mro__)

# 7 捕获所有的异常
try:
    pass
except Exception as e:
    print(e)


# 8 创建自定义的异常
class NetworkError(Exception):
    pass


class HostnameError(Exception):
    pass


class TimeoutError(Exception):
    pass


class ProtocolError(Exception):
    pass


class CustomError(Exception):
    def __init__(self, message, status):
        super().__init__(message, status)
        self.message = message
        self.status = status


# 9 通过引发异常来响应另一个异常
# explicit chaining.  Use this whenever your
# intent is to raise a new exception in response to another
def example1():
    try:
        int('N/A')
    except ValueError as e:
        raise RuntimeError('A parsing error occurred') from e


# implicit chaining.  This occurs if there's an
# unexpected exception in the except block.
def example2():
    try:
        int('N/A')
    except ValueError as e:
        print('It failed. Reason:', err)  # Intentional error


# discarding the previous exception
def example3():
    try:
        int('N/A')
    except ValueError as e:
        raise RuntimeError('A parsing error occurred') from None


if __name__ == '__main__':
    import traceback


    print('****** EXPLICIT EXCEPTION CHAINING ******')
    try:
        example1()
    except Exception:
        traceback.print_exc()

    print()
    print('****** IMPLICIT EXCEPTION CHAINING ******')
    try:
        example2()
    except Exception:
        traceback.print_exc()

    print()
    print('****** DISCARDED CHAINING *******')
    try:
        example3()
    except Exception:
        traceback.print_exc()

# 10 重新抛出上一个异常
try:
    pass
except ValueError:
    print("didn't work")
    raise

# 11 发出警告信息
import warnings


warnings.simplefilter('always')

warnings.warn('argument deprecated', DeprecationWarning)

f = open('data/data.json')
del f


# 12 对基本的程序崩溃问题进行调试
def func(n):
    return n + 10


try:
    func('hello')
except:
    traceback.print_exc(file=sys.stderr)

# pdb.pm() # 加载Python调试器

# 13 对程序做性能分析以及计时统计
import time
from functools import wraps


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r

    return wrapper


if __name__ == '__main__':
    @timethis
    def countdown(n):
        while n > 0:
            n -= 1


    countdown(10000000)

    from timeit import timeit


    print(timeit('math.sqrt(2)', 'import math', number=1000000))


# 14 让你的程序运行的更快
# 使用函数,涉及局部变量的操作要更快
def main(filename):
    with open(filename) as f:
        pass

# 有选择性的消除属性访问
# from module import name 以及选择性使用bound method避免出现属性查询操作

# 理解变量所处的位置,局部变量快

# 避免不必要的抽象,比如decorator,property,descriptor

# 使用内建的容器

# 避免产生不必要的数据结构或者拷贝动作
