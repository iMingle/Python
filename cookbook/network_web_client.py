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
TEST_UDP = True
if TEST_UDP:
    s = socket(AF_INET, SOCK_DGRAM)
    s.sendto(b'', ('localhost', 20000))
    resp = s.recvfrom(8192)
    print('UDP Response:', resp)
    s.close()
