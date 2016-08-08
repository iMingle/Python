"""Network and Web Programming.

"""

from socket import socket, AF_INET, SOCK_STREAM, SOCK_DGRAM

# TCP
TEST_TCP = False

if TEST_TCP:
    s = socket(AF_INET, SOCK_STREAM)
    s.connect(('localhost', 20000))

    s.send(b'Hello\n')
    resp = s.recv(8192)
    print('TCP Response:', resp)
    s.close()

# UDP
TEST_UDP = False
if TEST_UDP:
    s = socket(AF_INET, SOCK_DGRAM)
    s.sendto(b'', ('localhost', 20000))
    resp = s.recvfrom(8192)
    print('UDP Response:', resp)
    s.close()

# REST
TEST_REST = False
if TEST_REST:
    from urllib.request import urlopen

    u = urlopen('http://localhost:8080/hello?name=Guido')
    print(u.read().decode('utf-8'))

    u = urlopen('http://localhost:8080/localtime')
    print(u.read().decode('utf-8'))

# XML-RPC
TEST_XML_RPC = False
if TEST_XML_RPC:
    from xmlrpc.client import ServerProxy
    s = ServerProxy('http://localhost:15000', allow_none=True)
    s.set('foo', 'bar')
    s.set('spam', [1, 2, 3])
    print(s.keys())
    print(s.get('foo'))
    print(s.get('spam'))
    s.delete('spam')
    print(s.exists('spam'))

# 不同解释器间通信
TEST_MULTI_PROC = False
if TEST_MULTI_PROC:
    from multiprocessing.connection import Client
    c = Client(('localhost', 25000), authkey=b'peekaboo')
    c.send(b'hello')
    print(c.recv())
    c.send(42)
    print(c.recv())
    c.send([1, 2, 3, 4, 5])
    print(c.recv())
