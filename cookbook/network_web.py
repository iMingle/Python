"""Network and Web Programming.

"""

#1 以客户端的形式同HTTP服务交互
from urllib import request, parse

url = 'http://httpbin.org/get'
params = {
    'name1': 'value1',
    'name2': 'value2'
}
querystring = parse.urlencode(params)
u = request.urlopen(url + '?' + querystring)
resp = u.read()
print(resp)

url = 'http://httpbin.org/post'
params = {
    'name1': 'value1',
    'name2': 'value2'
}
querystring = parse.urlencode(params)
u = request.urlopen(url, querystring.encode('ascii'))
resp = u.read()
print(resp)

headers = {
    'User-agent': 'none/ofyourbudiness',
    'Spam': 'eggs'
}
req = request.Request(url, querystring.encode('ascii'), headers=headers)
u = request.urlopen(req)
resp = u.read()
print(resp)

# 复杂请求
import requests

url = 'http://httpbin.org/post'
params = {
    'name1': 'value1',
    'name2': 'value2'
}
headers = {
    'User-agent': 'none/ofyourbudiness',
    'Spam': 'eggs'
}
resp = requests.post(url, data=params, headers=headers)
print(resp.text)
print(resp.content)
print(resp.json)

# head请求
resp = requests.head('http://www.python.org/index.html')
status = resp.status_code
print(resp.headers)
print(status)
date = resp.headers['Date']
print(date)
location = resp.headers['Location']
print(location)
content_length = resp.headers['Content-Length']
print(content_length)

# 认证
resp = requests.get('http://pypi.python.org/pypi?:action=login', auth=('user', 'password'))
print(resp)

# 传递cookies
resp1 = requests.get(url)
resp2 = requests.get(url, cookies=resp1.cookies)
print(resp2)

# 文件上传
url = 'http://httpbin.org/post'
files = {
    'file': ('data/data.json', open('data/data.json', 'rb'))
}
r = requests.post(url, files=files)

# 标准库实现的HEAD请求
from http.client import HTTPConnection

c = HTTPConnection('www.python.org', 80)
c.request('HEAD', '/index.html')
resp = c.getresponse()
print('status', resp.status)
for name, value in resp.getheaders():
    print(name, value)

# 认证
import urllib

try:
    auth = urllib.request.HTTPBasicAuthHandler()
    auth.add_password('pypi', 'http://pypi.python.org', 'username', 'password')
    opener = urllib.request.build_opener(auth)
    r = urllib.request.Request('http://pypi.python.org/pypi?:action=login')
    u = opener.open(r)
    resp = u.read()
    print(resp)
except urllib.error.HTTPError as e:
    print(e)

#2 创建一个TCP服务器
from socketserver import BaseRequestHandler, StreamRequestHandler, TCPServer, ThreadingTCPServer
from socket import socket, AF_INET, SOCK_STREAM

class EchoHandler(BaseRequestHandler):
    def handle(self):
        print('>got connection from', self.client_address)
        while True:
            msg = self.request.recv(8192)
            if not msg:
                break
            self.request.send(msg)

TEST_TCP = False

if __name__ == '__main__' and TEST_TCP:
    TCPServer.allow_reuse_address = True
    # serv = TCPServer(('', 20000), EchoHandler)
    print('>echo server running on port 20000')
    serv = ThreadingTCPServer(('', 20000), EchoHandler)
    serv.serve_forever()

class EchoHandler(StreamRequestHandler):
    def handle(self):
        print('>>got connection from', self.client_address)
        # self.rfile is a file-like object for reading
        for line in self.rfile:
            self.wfile.write(line)

if __name__ == '__main__' and TEST_TCP:
    TCPServer.allow_reuse_address = True
    print('>>echo server running on port 20000')
    serv = ThreadingTCPServer(('', 20000), EchoHandler)
    serv.serve_forever()

# 多线程方式
if __name__ == '__main__' and TEST_TCP:
    from threading import Thread
    import socket

    NWORKERS = 16
    print('>>>echo server running on port 20000')
    serv = TCPServer(('', 20000), EchoHandler, bind_and_activate=False)
    # set up various socket options
    serv.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

    for n in range(NWORKERS):
        t = Thread(target=serv.serve_forever)
        t.daemon = True
        t.start()
    # bind and activate
    serv.server_bind()
    serv.server_activate()
    serv.serve_forever()

class EchoHandler(StreamRequestHandler):
    timeout = 5
    rbufsize = -1
    wbufsize = 0
    disable_nagle_algorithm = False
    def handle(self):
        print('>>>got connection from', self.client_address)
        # self.rfile is a file-like object for reading
        try:
            for line in self.rfile:
                # self.wfile is a file-like object for writing
                self.wfile.write(line)
        except socket.timeout:
            print('Timed out!')

if __name__ == '__main__' and TEST_TCP:
    TCPServer.allow_reuse_address = True
    serv = TCPServer(('', 20000), EchoHandler)
    print('>>>>echo server running on port 20000')
    serv.serve_forever()

def echo_handler(address, client_sock):
    print('got connection from {}'.format(address))
    while True:
        msg = client_sock.recv(8192)
        if not msg:
            break
        client_sock.sendall(msg)
    client_sock.close()

def echo_server(address, backlog=5):
    sock = socket(AF_INET, SOCK_STREAM)
    sock.bind(address)
    sock.listen(backlog)
    while True:
        client_sock, client_addr = sock.accept()
        echo_handler(client_addr, client_sock)

if __name__ == '__main__' and TEST_TCP:
    TCPServer.allow_reuse_address = True
    print('>>>>>echo server running on port 20000')
    echo_server(('', 20000))

#3 创建一个UDP服务器
from socketserver import BaseRequestHandler, UDPServer, ThreadingUDPServer
import time

class TimeHandler(BaseRequestHandler):
    def handle(self):
        print('got connection from', self.client_address)
        # Get message and client socket
        msg, sock = self.request
        resp = time.ctime()
        sock.sendto(resp.encode('ascii'), self.client_address)

TEST_UDP = False

if __name__ == '__main__' and TEST_UDP:
    UDPServer.allow_reuse_address = True
    # serv = UDPServer(('', 20000), TimeHandler)
    serv = ThreadingUDPServer(('', 20000), TimeHandler)
    serv.serve_forever()

from socket import socket, AF_INET, SOCK_DGRAM

def time_server(address):
    sock = socket(AF_INET, SOCK_DGRAM)
    sock.bind(address)
    while True:
        msg, addr = sock.recvfrom(8192)
        print('got message from', addr)
        resp = time.ctime()
        sock.sendto(resp.encode('ascii'), addr)

if __name__ == '__main__' and TEST_UDP:
    UDPServer.allow_reuse_address = True
    time_server(('', 20000))

#4 从CIDR(Classless InterDomain Routing)地址中生成IP地址的范围
import ipaddress

net = ipaddress.ip_network('123.45.67.64/27')
print(net)
for a in net:
    print(a)

net6 = ipaddress.ip_network('12:3456:78:90ab:cd:ef01:23:30/125')
print(net6)
for a in net6:
    print(a)
