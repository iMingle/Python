"""简单的web服务

"""

from http.server import SimpleHTTPRequestHandler
from http.server import HTTPServer
from socketserver import ThreadingMixIn
import math


PORT = 8080


class SimpleHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        summary = 0
        for i in range(1000000):
            summary += math.sqrt(i)
        print(summary)
        SimpleHTTPRequestHandler.do_GET(self)
        return


class ThreadingHttpServer(ThreadingMixIn, HTTPServer):
    pass


handler = SimpleHandler

with ThreadingHttpServer(("", PORT), handler) as httpd:
    print("serving at port", PORT)
    httpd.serve_forever()
