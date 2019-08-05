"""爬虫下载图片

"""
import requests
import json
from selenium import webdriver
from lxml import etree


def download(src, id, postfix):
    dir = './' + str(id) + postfix
    try:
        pic = requests.get(src, timeout=10)
        fp = open(dir, 'wb')
        fp.write(pic.content)
        fp.close()
    except requests.exceptions.ConnectionError:
        print('图片无法下载')


def query_photo(query):
    for i in range(0, 40, 10):
        url = 'https://www.douban.com/j/search_photo?q=' + query + '&limit=20&start=40'
        html = requests.get(url).text
        response = json.loads(html, encoding='utf-8')
        for image in response['images']:
            print(image['src'])
            download(image['src'], image['id'], '.jpg')


def download_xpath(query):
    request_url = 'https://movie.douban.com/subject_search?search_text=' + query + '&cat=1002'
    driver = webdriver.Chrome(executable_path='../autotest/data/chromedriver')
    driver.get(request_url)
    src_xpath = "//div[@class='item-root']/a[@class='cover-link']/img[@class='cover']/@src"
    title_xpath = "//div[@class='item-root']/div[@class='detail']/div[@class='title']/a[@class='title-text']"
    html = etree.HTML(driver.page_source)
    srcs = html.xpath(src_xpath)
    titles = html.xpath(title_xpath)
    for src, title in zip(srcs, titles):
        download(src, title.text, '.webp')


if __name__ == '__main__':
    # query_photo('赫本')
    download_xpath('赫本')
