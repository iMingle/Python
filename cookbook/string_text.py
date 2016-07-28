"""Strings and Text.

"""

#1 针对任意多的分隔符拆分字符串
line = "mingle:26:jinminglei@yeah.net"
split = line.split(":")
print(split)

import re # 正则表达式 

line = "asdf fjdk; afed, fjek,asdf,      foo"
resplit = re.split(r"[;,\s]\s*", line)
print(resplit)
fields = re.split(r"(;|,|\s)\s*", line)
print(fields)
values = fields[::2]
delimiters = fields[1::2] + [""]
print(values)
print(delimiters)
print("".join(v+d for v, d in zip(values, delimiters)))

# 非捕获组
print(re.split(r"(?:;|,|\s)\s*", line))

#2 在字符串的开头或结尾处做文本匹配
filename = "file.txt"
print(filename.endswith(".txt"))
print(filename.startswith("file:"))
url = "http://www.python.org"
print(url.startswith("http:"))

import os

filenames = os.listdir(".")
print(filenames)
print([name for name in filenames if name.endswith((".c", ".py"))])
print(any(name.endswith(".py") for name in filenames))

from urllib.request import urlopen

def read_data(name):
	if name.startswith(("http:", "https:", "ftp:")):
		return urlopen(name).read()
	else:
		with open(name) as f:
			return f.read()

choices = ["http:", "ftp:"]
url = "http://www.python.org"
# print(url.startswith(choices)) # error
print(url.startswith(tuple(choices)))

#3 利用Shell通配符做字符串匹配
from fnmatch import fnmatch, fnmatchcase

print(fnmatch("foo.txt", "*.txt"))
print(fnmatch("foo.txt", "?oo.txt"))
print(fnmatch("Dat45.csv", "Dat[0-9]*"))
names = ["Dat1.csv", "Dat2.csv", "config.ini", "foo.py"]
print([name for name in names if fnmatch(name, "Dat*.csv")])

print(fnmatch("foo.txt", "*.TXT"))
print(fnmatchcase("foo.txt", "*.TXT"))

addresses = [
    '5412 N CLARK ST',
    '1060 W ADDISON ST',
    '1039 W GRANVILLE AVE',
    '2122 N CLARK ST',
    '4802 N BROADWAY',
]

a = [addr for addr in addresses if fnmatchcase(addr, "* ST")]
print(a)
b = [addr for addr in addresses if fnmatchcase(addr, "54[0-9][0-9] *CLARK*")]
print(b)

#4 文本模式的匹配和查找
text = "yeah, but no, but yeah, but no, but yeah"
print("yeah" == text)
print(text.startswith("yeah"))
print(text.endswith("no"))
print(text.find("no"))

text1 = "11/27/2016"
text2 = "Nov 27, 2016"
if re.match(r"\d+/\d+/\d+", text1):
	print("yes")
else:
	print("no")
if re.match(r"\d+/\d+/\d+", text2):
	print("yes")
else:
	print("no")

datepat = re.compile(r"\d+/\d+/\d+")
if datepat.match(text1):
	print("yes")
else:
	print("no")
if datepat.match(text2):
	print("yes")
else:
	print("no")

text = "Today is 11/27/2016. PyCon starts 3/13/2013."
print(datepat.findall(text))

datepat = re.compile(r"(\d+)/(\d+)/(\d+)")
m = datepat.match("11/27/2016")
print(m)
print(m.group(1))
print(m.group(2))
print(m.group(3))
print(m.groups())
month, day, year = m.groups()
print(datepat.findall(text))
for month, day, year in datepat.findall(text):
	print("{}-{}-{}".format(year, month, day))

for m in datepat.finditer(text):
	print(m.groups())

#5 查找和替换文本
text = "yeah, but no, but yeah, but no, but yeah"
print(text.replace("yeah", "yep"))

text = "Today is 11/27/2016. PyCon starts 3/13/2013."
print(re.sub(r"(\d+)/(\d+)/(\d+)", r"\3-\1-\2", text))
datepat = re.compile(r"(\d+)/(\d+)/(\d+)")
print(datepat.sub(r"\3-\1-\2", text))

from calendar import month_abbr

def change_date(m):
	mon_name = month_abbr[int(m.group(1))]
	return "{} {} {}".format(m.group(2), mon_name, m.group(3))
print(datepat.sub(change_date, text))

newtext, n = datepat.subn(r"\3-\1-\2", text)
print(newtext)
print(n)

#6 以不区分大小写的方式对文本做查找和替换
text = "UPPER PYTHON, lower python, Mixed Python"
print(re.findall("python", text))
print(re.findall("python", text, flags=re.IGNORECASE))
print(re.sub("python", "snake", text))
print(re.sub("python", "snake", text, flags=re.IGNORECASE))

def matchcase(word):
	"""替换的文本和匹配的文本大小写吻合"""
	def replace(m):
		text = m.group()
		if text.isupper():
			return word.upper()
		elif text.islower():
			return word.lower()
		elif text[0].isupper():
			return word.capitalize()
		else:
			return word
	return replace
print(re.sub("python", matchcase("snake"), text, flags=re.IGNORECASE))

#7 定义实现最短匹配的正则表达式
str_pat = re.compile(r"'(.*)'")
text1 = "Computer says 'no.'"
print(str_pat.findall(text1))
text2 = "Computer says 'no.' Phone says 'yes.'"
print(str_pat.findall(text2))

# ?表示非贪心模式
str_pat = re.compile(r"'(.*?)'")
print(str_pat.findall(text2))

#8 编写多行模式的正则表达式
# .不能匹配换行符
comment = re.compile(r"/\*(.*?)\*/")
text1 = "/* this is a comment */"
text2 = """/* this is a
	multiline comment */
"""
print(comment.findall(text1))
print(comment.findall(text2))

comment = re.compile(r"/\*((?:.|\n)*?)\*/")
print(comment.findall(text2))
comment = re.compile(r"/\*(.*?)\*/", re.DOTALL)
print(comment.findall(text2))

#9 将Unicode文本统一表示为规范形式
import io  
import sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8') #改变标准输出的默认编码
s1 = u"Spicy Jalape\u00f1o"
s2 = u"Spicy Jalapen\u0303o"
print(s1)
print(s2)
print(s1 == s2) # False

import unicodedata
t1 = unicodedata.normalize("NFC", s1)
t2 = unicodedata.normalize("NFC", s2)
print(t1 == t2) # True
print(ascii(t1))
t3 = unicodedata.normalize("NFD", s1)
t4 = unicodedata.normalize("NFD", s2)
print(t3 == t4) # True

s = "\ufb01"
print(s)
print(unicodedata.normalize("NFD", s))
print(unicodedata.normalize("NFC", s))
print(unicodedata.normalize("NFKD", s))
print(unicodedata.normalize("NFKC", s))

# 去除音符标记
t1 = unicodedata.normalize("NFD", s1)
print(s1)
print("".join(c for c in t1 if not unicodedata.combining(c)))

#10 用正则表达式处理Unicode字符
num = re.compile(r"\d+")
# ASCII 数字
print(num.match("123"))
# 阿拉伯数字
print(num.match("\u0661\u0662\u0663"))

pat = re.compile("stra\u00dfe", re.IGNORECASE)
s = "straße"
print(pat.match(s))
print(pat.match(s.upper()))
print(s.upper()) # STRASSE

#11 从字符串中去掉不需要的字符
# 去除空格
s = " hello world \n"
print(s.strip())
print(s.lstrip())
print(s.rstrip())

# 去除字符
t = "-----hello====="
print(t.lstrip("-"))
print(t.strip("-="))
print(t.rstrip("="))

#12 文本过滤和清理
# A tricky string
s = 'p\xfdt\u0125\xf6\xf1\x0cis\tawesome\r\n'
print(s)
remap = {
    ord('\t') : ' ',
    ord('\f') : ' ',
    ord('\r') : None      # Deleted
}

a = s.translate(remap)
print('whitespace remapped:', a)

# Unicode组合字符: None
cmb_chrs = dict.fromkeys(c for c in range(sys.maxunicode) if unicodedata.combining(chr(c)))
b = unicodedata.normalize("NFD", a)
print(b)
print(b.translate(cmb_chrs))

# Nd=Decimal_Number
digitmap = {c: ord("0") + unicodedata.digit(chr(c)) for c in range(sys.maxunicode) if unicodedata.category(chr(c)) == "Nd"}
print(len(digitmap))
x = "\u0661\u0662\u0663"
print(x.translate(digitmap))

b = unicodedata.normalize("NFD", a)
b.encode("ascii", "ignore").decode("ascii")
print(b)

#13 对齐文本字符串
text = "Hello World"
print(text.ljust(20))
print(text.rjust(20))
print(text.center(20))
print(text.ljust(20, "*"))
print(text.rjust(20, "*"))
print(text.center(20, "*"))

print(format(text, "=>20"))
print(format(text, "=<20"))
print(format(text, "=^20"))

print("{:>10s} {:>10s}".format("Hello", "World"))

x = 1.2345
print(format(x, ">10"))
print(format(x, "^10.2f"))

#14 字符串连接及合并
parts = ["Is", "Chicago", "Not", "Chicago?"]
print(" ".join(parts))
print(",".join(parts))
print("".join(parts))

a = "Is Chicago"
b = "Not Chicago?"
print(a + " " + b)
print("{} {}".format(a, b))
a = "Helo" "World"
print(a)
print(a, b, sep=":")

def sample():
    yield "Is"
    yield "Chicago"
    yield "Not"
    yield "Chicago?"
print(" ".join(sample()))

#15 给字符串中的变量名做插值处理
s = "{name} has {n} messages."
print(s.format(name="Mingle", n=20))
name = "Mingle"
n = 20
print(s.format_map(vars()))

class Info:
    def __init__(self, name, n):
        super(Info, self).__init__()
        self.name = name
        self.n = n
a = Info("Mingle", 20)
print(s.format_map(vars(a)))

# print(s.format(name="Mingle")) # error
class safesub(dict):
    def __missing__(self, key):
        return "{" + key + "}"
del n
print(s.format_map(safesub(vars())))

def sub(text):
    """变量替换"""
    return text.format_map(safesub(sys._getframe(1).f_locals))
print(sub("Hello {name}"))
n = 20
print(sub("You have {n} messages."))
print(sub("You have {notexisted} messages."))

#16 以固定的列数重新格式化文本
s = "Look into my eyes, look into my eyes, the eyes, the eyes, \
the eyes, not around the eyes, don't look around the eyes, \
look into my eyes, you're under."

import textwrap

print(textwrap.fill(s, 70))
print(textwrap.fill(s, 40))
print(textwrap.fill(s, 40, initial_indent=" "))
print(textwrap.fill(s, 40, subsequent_indent=" "))

import shutil
print(shutil.get_terminal_size())

#17 在文本中处理HTML和XML实体
s = "Elements are written as '<tag>text</tag>'."
import html
print(s)
print(html.escape(s))
print(html.escape(s, quote=False))

s = "Spicy Jalape\u00f1o"
print(s.encode("ascii", errors="xmlcharrefreplace"))

s = "Spicy &quot;Jalape&#241;o&quot."
from html.parser import HTMLParser
p = HTMLParser()
print(p.unescape(s))
t = "The prompt is &gt;&gt;&gt;"
from xml.sax.saxutils import unescape
print(unescape(t))

#18 文本分词
text = "foo = 23 + 42 * 10"

import re
from collections import namedtuple

# 命名捕获组
NAME = r"(?P<NAME>[a-zA-Z_][a-zA-Z_0-9]*)"
NUM  = r"(?P<NUM>\d+)"
PLUS = r"(?P<PLUS>\+)"
TIMES = r"(?P<TIMES>\*)"
EQ    = r"(?P<EQ>=)"
WS    = r"(?P<WS>\s+)"

master_pat = re.compile("|".join([NAME, NUM, PLUS, TIMES, EQ, WS]))

scanner = master_pat.scanner("foo = 42")
match = scanner.match()
print(match)
print(match.lastgroup, match.group(), sep=":")
match = scanner.match()
print(match.lastgroup, match.group(), sep=":")
match = scanner.match()
print(match.lastgroup, match.group(), sep=":")
match = scanner.match()
print(match.lastgroup, match.group(), sep=":")
match = scanner.match()
print(match.lastgroup, match.group(), sep=":")

Token = namedtuple("Token", ["type", "value"])
def generate_tokens(pat, text):
    scanner = pat.scanner(text)
    for m in iter(scanner.match, None):
        yield Token(m.lastgroup, m.group())
for tok in generate_tokens(master_pat, "foo = 42"):
    print(tok)

tokens = (tok for tok in generate_tokens(master_pat, text) if tok.type != "WS")
for tok in tokens:
    print(tok)

#19 编写一个简单的递归下降解析器
import collections

# Token specification
NUM    = r"(?P<NUM>\d+)"
PLUS   = r"(?P<PLUS>\+)"
MINUS  = r"(?P<MINUS>-)"
TIMES  = r"(?P<TIMES>\*)"
DIVIDE = r"(?P<DIVIDE>/)"
LPAREN = r"(?P<LPAREN>\()"
RPAREN = r"(?P<RPAREN>\))"
WS     = r"(?P<WS>\s+)"

master_pat = re.compile("|".join([NUM, PLUS, MINUS, TIMES, 
                                  DIVIDE, LPAREN, RPAREN, WS]))

# Tokenizer
Token = collections.namedtuple("Token", ["type","value"])

def generate_tokens(text):
    scanner = master_pat.scanner(text)
    for m in iter(scanner.match, None):
        tok = Token(m.lastgroup, m.group())
        if tok.type != "WS":
            yield tok

# Parser 
class ExpressionEvaluator:
    """
    Implementation of a recursive descent parser.   Each method
    implements a single grammar rule.  Use the ._accept() method
    to test and accept the current lookahead token.  Use the ._expect()
    method to exactly match and discard the next token on on the input
    (or raise a SyntaxError if it doesn't match).
    """

    def parse(self,text):
        self.tokens = generate_tokens(text)
        self.tok = None             # Last symbol consumed
        self.nexttok = None         # Next symbol tokenized
        self._advance()             # Load first lookahead token
        return self.expr()

    def _advance(self):
        "Advance one token ahead"
        self.tok, self.nexttok = self.nexttok, next(self.tokens, None)

    def _accept(self,toktype):
        "Test and consume the next token if it matches toktype"
        if self.nexttok and self.nexttok.type == toktype:
            self._advance()
            return True
        else:
            return False

    def _expect(self,toktype):
        "Consume next token if it matches toktype or raise SyntaxError"
        if not self._accept(toktype):
            raise SyntaxError('Expected ' + toktype)

    # Grammar rules follow

    def expr(self):
        "expression ::= term { ('+'|'-') term }*"

        exprval = self.term()
        while self._accept("PLUS") or self._accept("MINUS"):
            op = self.tok.type
            right = self.term()
            if op == "PLUS":
                exprval += right
            elif op == "MINUS":
                exprval -= right
        return exprval
    
    def term(self):
        "term ::= factor { ('*'|'/') factor }*"

        termval = self.factor()
        while self._accept("TIMES") or self._accept("DIVIDE"):
            op = self.tok.type
            right = self.factor()
            if op == "TIMES":
                termval *= right
            elif op == "DIVIDE":
                termval /= right
        return termval

    def factor(self):
        "factor ::= NUM | ( expr )"

        if self._accept("NUM"):
            return int(self.tok.value)
        elif self._accept("LPAREN"):
            exprval = self.expr()
            self._expect("RPAREN")
            return exprval
        else:
            raise SyntaxError("Expected NUMBER or LPAREN")

if __name__ == "__main__":
    e = ExpressionEvaluator()
    print(e.parse("2"))
    print(e.parse("2 + 3"))
    print(e.parse("2 + 3 * 4"))
    print(e.parse("2 + (3 + 4) * 5"))

# Example of building trees
class ExpressionTreeBuilder(ExpressionEvaluator):
    def expr(self):
        "expression ::= term { ('+'|'-') term }"

        exprval = self.term()
        while self._accept("PLUS") or self._accept("MINUS"):
            op = self.tok.type
            right = self.term()
            if op == "PLUS":
                exprval = ("+", exprval, right)
            elif op == "MINUS":
                exprval = ("-", exprval, right)
        return exprval
    
    def term(self):
        "term ::= factor { ('*'|'/') factor }"

        termval = self.factor()
        while self._accept("TIMES") or self._accept("DIVIDE"):
            op = self.tok.type
            right = self.factor()
            if op == "TIMES":
                termval = ("*", termval, right)
            elif op == "DIVIDE":
                termval = ("/", termval, right)
        return termval

    def factor(self):
        "factor ::= NUM | ( expr )"

        if self._accept("NUM"):
            return int(self.tok.value)
        elif self._accept("LPAREN"):
            exprval = self.expr()
            self._expect("RPAREN")
            return exprval
        else:
            raise SyntaxError("Expected NUMBER or LPAREN")

if __name__ == "__main__":
    e = ExpressionTreeBuilder()
    print(e.parse("2 + 3"))
    print(e.parse("2 + 3 * 4"))
    print(e.parse("2 + (3 + 4) * 5"))
    print(e.parse("2 + 3 + 4"))

# parse by ply
from ply.lex import lex
from ply.yacc import yacc

# Token list
tokens = [ "NUM", "PLUS", "MINUS", "TIMES", "DIVIDE", "LPAREN", "RPAREN" ]

# Ignored characters

t_ignore = " \t\n"

# Token specifications (as regexs)
t_PLUS   = r"\+"
t_MINUS  = r"-"
t_TIMES  = r"\*"
t_DIVIDE = r"/"
t_LPAREN = r"\("
t_RPAREN = r"\)"

# Token processing functions
def t_NUM(t):
    r"\d+"
    t.value = int(t.value)
    return t

# Error handler
def t_error(t):
    print("Bad character: {!r}".format(t.value[0]))
    t.skip(1)

# Build the lexer
lexer = lex()

# Grammar rules and handler functions
def p_expr(p):
    """
    expr : expr PLUS term
         | expr MINUS term
    """
    if p[2] == "+":
        p[0] = p[1] + p[3]
    elif p[2] == "-":
        p[0] = p[1] - p[3]

def p_expr_term(p):
    """
    expr : term
    """
    p[0] = p[1]

def p_term(p):
    """
    term : term TIMES factor
         | term DIVIDE factor
    """
    if p[2] == "*":
        p[0] = p[1] * p[3]
    elif p[2] == "/":
        p[0] = p[1] / p[3]

def p_term_factor(p):
    """
    term : factor
    """
    p[0] = p[1]

def p_factor(p):
    """
    factor : NUM
    """
    p[0] = p[1]

def p_factor_group(p):
    """
    factor : LPAREN expr RPAREN
    """
    p[0] = p[2]

def p_error(p):
    print("Syntax error")

parser = yacc()

if __name__ == "__main__":
    print(parser.parse("2"))
    print(parser.parse("2 + 3"))
    print(parser.parse("2 + (3 + 4) * 5"))

#20 在字节串上执行文本操作
data = b"Hello World"
print(data[0:5])
print(data.startswith(b"Hello"))
print(data.split())
print(data.replace(b"Hello", b"Hello Cruel"))

data = bytearray(b"Hello World")
print(data[0:5])
print(data.startswith(b"Hello"))
print(data.split())
print(data.replace(b"Hello", b"Hello Cruel"))

data = b"FOO:BAR,SPAM"
print(re.split(b"[:,]", data))
print(data[0])
print(data.decode("ascii"))
