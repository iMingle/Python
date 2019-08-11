"""K-Means
K-Means 是一种非监督学习，解决的是聚类问题。K 代表的是 K 类，Means 代表的是中心，
你可以理解这个算法的本质是确定 K 类的中心点，当你找到了这些中心点，也就完成了聚类。
"""
import PIL.Image as image
import numpy as np
import pandas as pd
from skimage import color
from sklearn import preprocessing
from sklearn.cluster import KMeans


# 输入数据
data = pd.read_csv('./data/kmeans_data.csv')
train_x = data[["2019年国际排名", "2018世界杯", "2015亚洲杯"]]
df = pd.DataFrame(train_x)
kmeans = KMeans(n_clusters=3)
# 规范化到 [0,1] 空间
min_max_scaler = preprocessing.MinMaxScaler()
train_x = min_max_scaler.fit_transform(train_x)
# kmeans 算法
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
# 合并聚类结果，插入到原数据中
result = pd.concat((data, pd.DataFrame(predict_y)), axis=1)
result.rename({0: u'聚类'}, axis=1, inplace=True)
print(result)


def load_data(file_path):
    """加载图像，并对数据进行规范化
    :param file_path:
    :return:
    """
    # 读文件
    f = open(file_path, 'rb')
    image_data = []
    # 得到图像的像素值
    img = image.open(f)
    # 得到图像尺寸
    width, height = img.size
    for x in range(width):
        for y in range(height):
            # 得到点 (x,y) 的三个通道值
            c1, c2, c3 = img.getpixel((x, y))
            image_data.append([c1, c2, c3])
    f.close()
    # 采用 Min-Max 规范化
    mm = preprocessing.MinMaxScaler()
    image_data = mm.fit_transform(image_data)
    return np.mat(image_data), width, height


# 加载图像，得到规范化的结果 img，以及图像尺寸
img, width, height = load_data('./data/weixin.jpg')

# 用 K-Means 对图像进行 2 聚类
kmeans = KMeans(n_clusters=2)
kmeans.fit(img)
label = kmeans.predict(img)
# 将图像聚类结果，转化成图像尺寸的矩阵
label = label.reshape([width, height])
# 创建个新图像 pic_mark，用来保存图像聚类的结果，并设置不同的灰度值
pic_mark = image.new("L", (width, height))
for x in range(width):
    for y in range(height):
        # 根据类别设置图像灰度, 类别 0 灰度值为 255， 类别 1 灰度值为 127
        pic_mark.putpixel((x, y), int(256 / (label[x][y] + 1)) - 1)
pic_mark.save("./data/weixin_mark.jpg", "JPEG")

# 将聚类标识矩阵转化为不同颜色的矩阵
label_color = (color.label2rgb(label) * 255).astype(np.uint8)
label_color = label_color.transpose(1, 0, 2)
images = image.fromarray(label_color)
images.save('./data/weixin_mark_color.jpg')


def load_data_original(file_path):
    # 读文件
    f = open(file_path, 'rb')
    data = []
    # 得到图像的像素值
    img = image.open(f)
    # 得到图像尺寸
    width, height = img.size
    for x in range(width):
        for y in range(height):
            # 得到点 (x,y) 的三个通道值
            c1, c2, c3 = img.getpixel((x, y))
            data.append([(c1 + 1) / 256.0, (c2 + 1) / 256.0, (c3 + 1) / 256.0])
    f.close()
    return np.mat(data), width, height


# 生成原图
# 加载图像，得到规范化的结果 imgData，以及图像尺寸
img, width, height = load_data_original('./data/weixin.jpg')
# 用 K-Means 对图像进行 16 聚类
kmeans = KMeans(n_clusters=16)
label = kmeans.fit_predict(img)
# 将图像聚类结果，转化成图像尺寸的矩阵
label = label.reshape([width, height])
# 创建个新图像 img，用来保存图像聚类压缩后的结果
img = image.new('RGB', (width, height))
for x in range(width):
    for y in range(height):
        c1 = kmeans.cluster_centers_[label[x, y], 0]
        c2 = kmeans.cluster_centers_[label[x, y], 1]
        c3 = kmeans.cluster_centers_[label[x, y], 2]
        img.putpixel((x, y), (int(c1 * 256) - 1, int(c2 * 256) - 1, int(c3 * 256) - 1))
img.save('./data/weixin_new.jpg')
