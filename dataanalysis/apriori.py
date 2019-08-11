"""Apriori
关联规则算法
"""
import csv
import time

from efficient_apriori import apriori
from lxml import etree
from selenium import webdriver


# 设置数据集
data = [['牛奶', '面包', '尿布'],
        ['可乐', '面包', '尿布', '啤酒'],
        ['牛奶', '尿布', '啤酒', '鸡蛋'],
        ['面包', '牛奶', '尿布', '啤酒'],
        ['面包', '牛奶', '尿布', '可乐']]

# 挖掘频繁项集和频繁规则
itemsets, rules = apriori(data, min_support=0.5, min_confidence=1)
print(itemsets)
print(rules)


def download(request_url, director, csv_write):
    flags = []
    driver = webdriver.Chrome(executable_path='../autotest/data/chromedriver')
    driver.get(request_url)
    time.sleep(1)
    html = driver.find_element_by_xpath("//*").get_attribute("outerHTML")
    html = etree.HTML(html)
    # 设置电影名称，导演演员 的 XPATH
    movie_lists = html.xpath(
        "/html/body/div[@id='wrapper']/div[@id='root']/div[1]//div[@class='item-root']/div[@class='detail']/div[@class='title']/a[@class='title-text']")
    name_lists = html.xpath(
        "/html/body/div[@id='wrapper']/div[@id='root']/div[1]//div[@class='item-root']/div[@class='detail']/div[@class='meta abstract_2']")
    # 获取返回的数据个数
    num = len(movie_lists)
    if num > 15:  # 第一页会有 16 条数据
        # 默认第一个不是，所以需要去掉
        movie_lists = movie_lists[1:]
        name_lists = name_lists[1:]
    for (movie, name_list) in zip(movie_lists, name_lists):
        # 会存在数据为空的情况
        if name_list.text is None:
            continue
        # 显示下演员名称
        print(name_list.text)
        names = name_list.text.split('/')
        # 判断导演是否为指定的 director
        if names[0].strip() == director and movie.text not in flags:
            # 将第一个字段设置为电影名称
            names[0] = movie.text
            flags.append(movie.text)
            csv_write.writerow(names)
    print('OK')  # 代表这页数据下载成功
    print(num)
    if num >= 14:  # 有可能一页会有 14 个电影
        # 继续下一页
        return True
    else:
        # 没有下一页
        return False


def cycle_download(director):
    # 写 CSV 文件
    file_name = './data/' + director + '.csv'
    out = open(file_name, 'w', newline='', encoding='utf-8-sig')
    csv_write = csv.writer(out, dialect='excel')
    base_url = 'https://movie.douban.com/subject_search?search_text=' + director + '&cat=1002&start='
    # 开始的 ID 为 0，每页增加 15
    start = 0
    while start < 10000:  # 最多抽取 1 万部电影
        request_url = base_url + str(start)
        # 下载数据，并返回是否有下一页
        flag = download(request_url, director, csv_write)
        if flag:
            start = start + 15
        else:
            break
    out.close()
    print('finished')


def apriori_test():
    director = u'宁浩'
    file_name = './data/' + director + '.csv'
    lists = csv.reader(open(file_name, 'r', encoding='utf-8-sig'))
    # 数据加载
    data = []
    for names in lists:
        name_new = []
        for name in names:
            # 去掉演员数据中的空格
            name_new.append(name.strip())
        data.append(name_new[1:])
    # 挖掘频繁项集和关联规则
    itemsets, rules = apriori(data, min_support=0.4, min_confidence=1)
    print(itemsets)
    print(rules)


if __name__ == '__main__':
    # cycle_download(u'宁浩')
    apriori_test()
