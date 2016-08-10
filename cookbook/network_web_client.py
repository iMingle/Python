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

# 远程过程调用
TEST_RPC = False

if TEST_RPC:
    import pickle
    class RPCProxy:
        def __init__(self, connection):
            super(RPCProxy, self).__init__()
            self._connection = connection

        def __getattr__(self, name):
            def do_rpc(*args, **kwargs):
                self._connection.send(pickle.dumps((name, args, kwargs)))
                result = pickle.loads(self._connection.recv())
                if isinstance(result, Exception):
                    raise result
                return result
            return do_rpc

    from multiprocessing.connection import Client
    c = Client(('localhost', 17000), authkey=b'peekaboo')
    proxy = RPCProxy(c)
    print(proxy.add(2, 3))
    print(proxy.sub(2, 3))
    try:
        proxy.sub([1, 2], 4)
    except Exception as e:
        print(e)

# 以简单的方式验证客户端身份
import hmac
import os

def client_authenticate(connection, secret_key):
    """authenticate client to a remote service.
    connection represents a network connection.
    secret_key is a key known only to both client/server."""
    message = os.urandom(32)
    hash = hmac.new(secret_key, message)
    digest = hash.digest()
    connection.send(digest)

TEST_AUTH = False

if TEST_AUTH:
    secret_key = b'peekaboo'
    s = socket(AF_INET, SOCK_STREAM)
    s.connect(('localhost', 18000))
    client_authenticate(s, secret_key)
    s.send(b'Hello World')
    resp = s.recv(1024)
    print(resp)

