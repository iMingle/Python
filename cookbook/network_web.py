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

print(net.num_addresses)
print(net6[0])
print(net6[1])
print(net6[-1])
print(net6[-2])

a = ipaddress.ip_address('123.45.67.69')
print(a in net)
b = ipaddress.ip_address('123.45.67.123')
print(b in net)

inet = ipaddress.ip_interface('123.45.67.73/27')
print(inet.network)
print(inet.ip)

#5 创建基于REST风格的简单接口
import cgi

def notfound_404(environ, start_response):
    start_response('404 Not Found', [('Content-type', 'text/plain')])
    return [b'Not Found']

class PathDispatcher:
    def __init__(self):
        self.pathmap = {}

    def __call__(self, environ, start_response):
        path = environ['PATH_INFO']
        params = cgi.FieldStorage(environ['wsgi.input'], environ=environ)
        method = environ['REQUEST_METHOD'].lower()
        environ['params'] = {key: params.getvalue(key) for key in params}
        handler = self.pathmap.get((method, path), notfound_404)
        return handler(environ, start_response)

    def register(self, method, path, function):
        self.pathmap[method.lower(), path] = function
        return function

_hello_resp = """\
<html>
  <head>
     <title>Hello {name}</title>
   </head>
   <body>
     <h1>Hello {name}!</h1>
   </body>
</html>"""

def hello_world(environ, start_response):
    start_response('200 OK', [ ('Content-type','text/html')])
    params = environ['params']
    resp = _hello_resp.format(name=params.get('name'))
    yield resp.encode('utf-8')

_localtime_resp = """\
<?xml version="1.0"?>
<time>
  <year>{t.tm_year}</year>
  <month>{t.tm_mon}</month>
  <day>{t.tm_mday}</day>
  <hour>{t.tm_hour}</hour>
  <minute>{t.tm_min}</minute>
  <second>{t.tm_sec}</second>
</time>"""

def localtime(environ, start_response):
    start_response('200 OK', [ ('Content-type', 'application/xml') ])
    resp = _localtime_resp.format(t=time.localtime())
    yield resp.encode('utf-8')

TEST_REST = False

if __name__ == '__main__' and TEST_REST:
    # from resty import PathDispatcher
    from wsgiref.simple_server import make_server

    # create the dispatcher and register functions
    dispatcher = PathDispatcher()
    dispatcher.register('GET', '/hello', hello_world)
    dispatcher.register('GET', '/localtime', localtime)

    # launch a basic server
    httpd = make_server('', 8080, dispatcher)
    print('Serving on port 8080...')
    httpd.serve_forever()

#6 利用XML-RPC实现简单的远程过程调用
from xmlrpc.server import SimpleXMLRPCServer

class KeyValueServer:
    _rpc_methods_ = ['get', 'set', 'delete', 'exists', 'keys']
    def __init__(self, address):
        self._data = {}
        self._serv = SimpleXMLRPCServer(address, allow_none=True)
        for name in self._rpc_methods_:
            self._serv.register_function(getattr(self, name))

    def get(self, name):
        return self._data[name]

    def set(self, name, value):
        self._data[name] = value

    def delete(self, name):
        del self._data[name]

    def exists(self, name):
        return name in self._data

    def keys(self):
        return list(self._data)

    def serve_forever(self):
        self._serv.serve_forever()

TEST_XML_RPC = False

if __name__ == '__main__' and TEST_XML_RPC:
    kvserv = KeyValueServer(('', 15000))
    print('Serving on port 15000...')
    kvserv.serve_forever()

#7 在不同的解释器间进行通信
from multiprocessing.connection import Listener
import traceback

def echo_client(conn):
    try:
        while True:
            msg = conn.recv()
            conn.send(msg)
    except EOFError:
        print('connection closed')

def echo_server(address, authkey):
    serv = Listener(address, authkey=authkey)
    try:
        while True:
            client = serv.accept()
            echo_client(client)
    except Exception:
        traceback.print_exc()

TEST_MULTI_PROC = False

if TEST_MULTI_PROC:
    print('Serving on port 25000...')
    echo_server(('', 25000), authkey=b'peekaboo')

#8 实现远程过程调用
import pickle
from threading import Thread

class RPCHandler:
    """docstring for RPCHandler"""
    def __init__(self):
        self._functions = {}

    def register_function(self, func):
        self._functions[func.__name__] = func

    def handle_connection(self, connection):
        try:
            while True:
                # receive a message
                func_name, args, kwargs = pickle.loads(connection.recv())
                # run the rpc and send a response
                try:
                    r = self._functions[func_name](*args, **kwargs)
                    connection.send(pickle.dumps(r))
                except Exception as e:
                    connection.send(pickle.dumps(e))
        except EOFError:
            pass

def rpc_server(handler, address, authkey):
    sock = Listener(address, authkey=authkey)
    while True:
        client = sock.accept()
        t = Thread(target=handler.handle_connection, args=(client,))
        t.daemon = True
        t.start()

# some remote functions
def add(x, y):
    return x + y

def sub(x, y):
    return x - y

# register with a handler
handler = RPCHandler()
handler.register_function(add)
handler.register_function(sub)

# run the server
TEST_RPC = False

if TEST_RPC:
    print('Serving on port 17000...')
    rpc_server(handler, ('localhost', 17000), authkey=b'peekaboo')

#9 以简单的方式验证客户端身份
import hmac
import os

def server_authenticate(connection, secret_key):
    """request client authentication."""
    message = os.urandom(32)
    connection.send(message)
    hash = hmac.new(secret_key, message)
    digest = hash.digest()
    response = connection.recv(len(digest))
    return hmac.compare_digest(digest, response)

secret_key = b'peekaboo'
def echo_handler(client_sock):
    if not server_authenticate(client_sock, secret_key):
        client_sock.close()
        return
    while True:
        msg = client_sock.recv(8192)
        if not msg:
            break
        client_sock.sendall(msg)

def echo_server(address):
    s = socket(AF_INET, SOCK_STREAM)
    s.bind(address)
    s.listen(5)
    while True:
        c, a = s.accept()
        echo_handler(c)

TEST_AUTH = False

if TEST_AUTH:
    print('Serving on port 18000...')
    echo_server(('', 18000))

#10 为网络服务增加SSL支持
import ssl

KEYFILE = 'data/server_key.pem' # private key of the server
CERTFILE = 'data/server_cert.pem' # server certificate (given to client)

def echo_client(s):
    while True:
        data = s.recv(8192)
        if data == b'':
            break
        s.send(data)
    s.close()
    print('collention closed')

def echo_server(address):
    s = socket(AF_INET, SOCK_STREAM)
    s.bind(address)
    s.listen(1)
    # wrap with an SSL layer requiring client certs
    s_ssl = ssl.wrap_socket(s, keyfile=KEYFILE, certfile=CERTFILE, server_side=True)
    # wait for connections
    while True:
        try:
            c, a = s_ssl.accept()
            print('get connection', c, a)
            echo_client(c)
        except Exception as e:
            print('{}: {}'.format(e.__class__.__name__, e))

TEST_SSL = False

if TEST_SSL:
    print('Serving on port 20000...')
    echo_server(('', 20000))

class SSLMixin:
    """mixin class that support for SSL to existing servers based on the socketserver module."""
    def __init__(self, *args, keyfile=None, certfile=None, ca_certs=None, 
        cert_reqs=ssl.CERT_NONE, **kwargs):
        self._keyfile = keyfile
        self._certfile = certfile
        self._ca_certs = ca_certs
        self._cert_reqs = cert_reqs
        super().__init__(*args, **kwargs)

    def get_request(self):
        client, addr = super().get_request()
        client_ssl = ssl.wrap_socket(client,
                                     keyfile = self._keyfile,
                                     certfile = self._certfile,
                                     ca_certs = self._ca_certs,
                                     cert_reqs = self._cert_reqs,
                                     server_side = True)
        return client_ssl, addr

class SSLSimpleXMLRPCServer(SSLMixin, SimpleXMLRPCServer):
    pass

class KeyValueServer:
    _rpc_methods_ = ['get', 'set', 'delete', 'exists', 'keys']
    def __init__(self, *args, **kwargs):
        self._data = {}
        self._serv = SSLSimpleXMLRPCServer(*args, allow_none=True, **kwargs)
        for name in self._rpc_methods_:
            self._serv.register_function(getattr(self, name))

    def get(self, name):
        return self._data[name]

    def set(self, name, value):
        self._data[name] = value

    def delete(self, name):
        del self._data[name]

    def exists(self, name):
        return name in self._data

    def keys(self):
        return list(self._data)

    def serve_forever(self):
        self._serv.serve_forever()

TEST_SSL_HTTP = False

if TEST_SSL_HTTP:
    KEYFILE='data/server_key.pem'   # Private key of the server
    CERTFILE='data/server_cert.pem' # Server certificate
    CA_CERTS='data/client_cert.pem' # Certificates of accepted clients

    kvserv = KeyValueServer(('', 15000),
                            keyfile=KEYFILE,
                            certfile=CERTFILE,
                            ca_certs=CA_CERTS,
                            cert_reqs=ssl.CERT_REQUIRED)
    print('Serving on port 15000...')
    kvserv.serve_forever()



#11 在进程间传递socket文件描述符
from multiprocessing.reduction import recv_handle, send_handle
import socket

def worker(in_p, out_p):
    out_p.close()
    while True:
        fd = recv_handle(in_p)
        print('child: got fd', fd)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, fileno=fd) as s:
            while True:
                msg = s.recv(1024)
                if not msg:
                    break
                print('child: recv {!r}'.format(msg))
                s.send(msg)

def server(address, in_p, out_p, worker_pid):
    in_p.close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    s.bind(address)
    s.listen(1)
    while True:
        client, addr = s.accept()
        print('server: got connection from', addr)
        send_handle(out_p, client.fileno(), worker_pid)
        client.close()

import multiprocessing

if __name__ == '__main__' and False:
    c1, c2 = multiprocessing.Pipe()
    worker_p = multiprocessing.Process(target=worker, args=(c1, c2))
    worker_p.start()

    server_p = multiprocessing.Process(target=server, args=(('', 15000), c1, c2, worker_p.pid))
    server_p.start()

    c1.close();
    c2.close();

# 将服务器和工作者进程实现为完全分离的程序,分别启动
# server
def server(work_address, port):
    # wait for the worker to connect
    work_serv = Listener(work_address, authkey=b'peekaboo')
    worker = work_serv.accept()
    worker_pid = worker.recv()

    # now run a TCP/IP server and send clients to worker
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    s.bind(('', port))
    s.listen(1)
    while True:
        client, addr = s.accept()
        print('SERVER: Got connection from', addr)
        send_handle(worker, client.fileno(), worker_pid)
        client.close()
    
if __name__ == '__main__' and False:
    import sys
    if len(sys.argv) != 3:
        print('Usage1: server.py server_address port', file=sys.stderr)
        raise SystemExit(1)

    server(sys.argv[1], int(sys.argv[2]))

# worker
def worker(server_address):
    serv = Client(server_address, authkey=b'peekaboo')
    serv.send(os.getpid())
    while True:
        fd = recv_handle(serv)
        print('WORKER: GOT FD', fd)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, fileno=fd) as client:
            while True:
                msg = client.recv(1024)
                if not msg:
                    break
                print('WORKER: RECV {!r}'.format(msg))
                client.send(msg)
    
if __name__ == '__main__' and False:
    import sys
    if len(sys.argv) != 2:
        print('Usage1: worker.py server_address', file=sys.stderr)
        raise SystemExit(1)

    worker(sys.argv[1])

# 利用socket来传递文件描述符
def send_fd(sock, fd):
    '''
    Send a single file descriptor.
    '''
    sock.sendmsg([b'x'],
                 [(socket.SOL_SOCKET, socket.SCM_RIGHTS, struct.pack('i', fd))])
    ack = sock.recv(2)
    assert ack == b'OK'

def server(work_address, port):
    # Wait for the worker to connect
    work_serv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    work_serv.bind(work_address)
    work_serv.listen(1)
    worker, addr = work_serv.accept()

    # Now run a TCP/IP server and send clients to worker
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
    s.bind(('',port))
    s.listen(1)
    while True:
        client, addr = s.accept()
        print('SERVER: Got connection from', addr)
        send_fd(worker, client.fileno())
        client.close()
    
if __name__ == '__main__' and False:
    import sys
    if len(sys.argv) != 3:
        print('Usage2: server.py server_address port', file=sys.stderr)
        raise SystemExit(1)

    server(sys.argv[1], int(sys.argv[2]))

def recv_fd(sock):
    '''
    Receive a single file descriptor
    '''
    msg, ancdata, flags, addr = sock.recvmsg(1,
                                             socket.CMSG_LEN(struct.calcsize('i')))

    cmsg_level, cmsg_type, cmsg_data = ancdata[0]
    assert cmsg_level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS
    sock.sendall(b'OK')
    return struct.unpack('i', cmsg_data)[0]

import socket

def worker(server_address):
    serv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    serv.connect(server_address)
    while True:
        fd = recv_fd(serv)
        print('WORKER: GOT FD', fd)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM, fileno=fd) as client:
            while True:
                msg = client.recv(1024)
                if not msg:
                    break
                print('WORKER: RECV {!r}'.format(msg))
                client.send(msg)
    
if __name__ == '__main__' and False:
    import sys
    if len(sys.argv) != 2:
        print('Usage2: worker.py server_address', file=sys.stderr)
        raise SystemExit(1)

    worker(sys.argv[1])

#12 理解事件驱动型I/O
class EventHandler:
    def fileno(self):
        'Return the associated file descriptor'
        raise NotImplemented('must implement')

    def wants_to_receive(self):
        'Return True if receiving is allowed'
        return False

    def handle_receive(self):
        'Perform the receive operation'
        pass

    def wants_to_send(self):
        'Return True if sending is requested' 
        return False

    def handle_send(self):
        'Send outgoing data'
        pass

import select

def event_loop(handlers):
    while True:
        wants_recv = [h for h in handlers if h.wants_to_receive()]
        wants_send = [h for h in handlers if h.wants_to_send()]
        can_recv, can_send, _ = select.select(wants_recv, wants_send, [])
        for h in can_recv:
            h.handle_receive()
        for h in can_send:
            h.handle_send()

            import socket
import time

class UDPServer(EventHandler):
    def __init__(self, address):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(address)

    def fileno(self):
        return self.sock.fileno()

    def wants_to_receive(self):
        return True

class UDPTimeServer(UDPServer):
    def handle_receive(self):
        msg, addr = self.sock.recvfrom(1)
        self.sock.sendto(time.ctime().encode('ascii'), addr)

class UDPEchoServer(UDPServer):
    def handle_receive(self):
        msg, addr = self.sock.recvfrom(8192)
        self.sock.sendto(msg, addr)

TEST_EVENT_HANDLER_UDP = False

if __name__ == '__main__' and TEST_EVENT_HANDLER_UDP:
    handlers = [ UDPTimeServer(('',14000)), UDPEchoServer(('',15000))  ]
    event_loop(handlers)

class TCPServer(EventHandler):
    def __init__(self, address, client_handler, handler_list):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)
        self.sock.bind(address)
        self.sock.listen(1)
        self.client_handler = client_handler
        self.handler_list = handler_list

    def fileno(self):
        return self.sock.fileno()

    def wants_to_receive(self):
        return True

    def handle_receive(self):
        client, addr = self.sock.accept()
        # Add the client to the event loop's handler list
        self.handler_list.append(self.client_handler(client, self.handler_list))

class TCPClient(EventHandler):
    def __init__(self, sock, handler_list):
        self.sock = sock
        self.handler_list = handler_list
        self.outgoing = bytearray()

    def fileno(self):
        return self.sock.fileno()

    def close(self):
        self.sock.close()
        # Remove myself from the event loop's handler list
        self.handler_list.remove(self)
        
    def wants_to_send(self):
        return True if self.outgoing else False

    def handle_send(self):
        nsent = self.sock.send(self.outgoing)
        self.outgoing = self.outgoing[nsent:]

class TCPEchoClient(TCPClient):
    def wants_to_receive(self):
        return True
    
    def handle_receive(self):
        data = self.sock.recv(8192)
        if not data:
            self.close()
        else:
            self.outgoing.extend(data)

TEST_EVENT_HANDLER_TCP = False

if __name__ == '__main__' and TEST_EVENT_HANDLER_TCP:
   handlers = []
   handlers.append(TCPServer(('',16000), TCPEchoClient, handlers))
   event_loop(handlers)

from concurrent.futures import ThreadPoolExecutor

class ThreadPoolHandler(EventHandler):
    def __init__(self, nworkers):
        self.signal_done_sock, self.done_sock = socket.socketpair()
        self.pending = []
        self.pool = ThreadPoolExecutor(nworkers)

    def fileno(self):
        return self.done_sock.fileno()

    # Callback that executes when the thread is done
    def _complete(self, callback, r):
        self.pending.append((callback, r.result()))
        self.signal_done_sock.send(b'x')

    # Run a function in a thread pool
    def run(self, func, args=(), kwargs={},*,callback):
        r = self.pool.submit(func, *args, **kwargs)
        r.add_done_callback(lambda r: self._complete(callback, r))

    def wants_to_receive(self):
        return True

    # Run callback functions of completed work
    def handle_receive(self):
        # Invoke all pending callback functions 
        for callback, result in self.pending:
            callback(result)
            self.done_sock.recv(1)
        self.pending = []

# A really bad fibonacci implementation
def fib(n):
    if n < 2:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)

class UDPServer(EventHandler):
    def __init__(self, address):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(address)

    def fileno(self):
        return self.sock.fileno()

    def wants_to_receive(self):
        return True
    
class UDPFibServer(UDPServer):
    def handle_receive(self):
        msg, addr = self.sock.recvfrom(128)
        n = int(msg)
        pool.run(fib, (n,), callback=lambda r: self.respond(r, addr))

    def respond(self, result, addr):
        self.sock.sendto(str(result).encode('ascii'), addr)

TEST_EVENT_HANDLER_THREAD = False

if __name__ == '__main__' and TEST_EVENT_HANDLER_THREAD:
    pool = ThreadPoolHandler(16)
    handlers = [pool, UDPFibServer(('', 16000))]
    event_loop(handlers)

#13 发送和接受大型数组
# 利用memoryview对大型数组进行发送和接受
def send_from(arr, dest):
    view = memoryview(arr).cast('B') # 转为无符号字节
    while len(view):
        nsent = dest.send(view)
        view = view[nsent:]

TEST_BIG_ARR = True

if TEST_BIG_ARR:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 25000))
    s.listen(1)
    c, a = s.accept()
    import numpy
    a = numpy.arange(0.0, 50000000.0)
    send_from(a, c)
    c.close()
