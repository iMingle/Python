# Copyright (c) 2016, Mingle. All rights reserved.
# Author: Mingle
# Contact: jinminglei@yeah.net

import sys, warnings, os, platform, logging
import json
import urllib, urllib.parse, urllib.request, urllib.response

print(sys.version_info)
if sys.version_info[0] < 3:
	warnings.warn("Need Python 3.0 for this program to run", RuntimeWarning)
else:
	print("Proceed as normal")

if platform.platform().startswith("Windows"):
	logging_file = os.path.join(os.getenv("HOMEDRIVE"), "test.log")
else:
	logging_file = os.path.join(os.getenv("HOME"), "test.log")

logging.basicConfig(
	level = logging.DEBUG,
	format = "%(asctime)s : %(levelname)s : %(message)s",
	filename = logging_file,
	filemode = "w",
)

logging.debug("Start of the program")
logging.info("Doing something")
logging.warning("Dying now")

#Get your own APP ID at http://developer.yahoo.com/wsregapp/
SEARCH_BASE = 'http://apis.baidu.com/heweather/pro/attractions'

class YahooSearchError(Exception):
	pass

def search(query, results=20, start=1, **kwargs):
	kwargs.update({
		"query": query,
		"results": results,
		"start": start,
		"output": "json",
	})
	url = SEARCH_BASE + "?" + urllib.parse.urlencode(kwargs)
	result = json.load(urllib.request.urlopen(url))
	if "Error" in result:
		raise YahooSearchError(result["Error"])
	return result["ResultSet"]

query = input("What do you want to search for?")
for result in search(query)["Result"]:
	print("{0}: {1}".format(result["Title"], result["Url"]))
