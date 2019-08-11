"""Standard lib

"""

import sys, warnings, os, platform, logging
import json
import urllib, urllib.parse, urllib.request, urllib.response

print(sys.version_info)
if sys.version_info[0] < 3:
    warnings.warn('Need Python 3.0 for this program to run', RuntimeWarning)
else:
    print('Proceed as normal')

if platform.platform().startswith('Windows'):
    logging_file = os.path.join(os.getenv('HOMEDRIVE'), 'test.log')
else:
    logging_file = os.path.join(os.getenv('HOME'), 'test.log')

logging.basicConfig(
    level = logging.DEBUG,
    format = '%(asctime)s : %(levelname)s : %(message)s',
    filename = logging_file,
    filemode = 'w',
)

logging.debug('Start of the program')
logging.info('Doing something')
logging.warning('Dying now')

response = urllib.request.urlopen('https://www.baidu.com')
print(response.readline())
print(json.dumps(['foo', {'bar': ('baz', None, 1.0, 2)}]))
print(json.loads("['foo', {'bar':['baz', null, 1.0, 2]}]"))