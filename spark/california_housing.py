"""处理加州房屋信息,预测房价

数据地址:http://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

"""
import os

from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import DenseVector
from pyspark.ml.regression import LinearRegression
from pyspark.sql import Row
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType


os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"


def convert_column(df, names, newType):
    """
    列数据的类型转化

    :param df: data frame
    :param names: 列名称
    :param newType: 新数据类型
    :return: 转化后的data frame
    """
    for name in names:
        df = df.withColumn(name, df[name].cast(newType))
    return df


if __name__ == "__main__":
    # 初始化 SparkSession 和 SparkContext
    spark = SparkSession.builder.master("local").appName("California Housing") \
        .config("spark.executor.memory", "1gb").getOrCreate()
    sc = spark.sparkContext

    # 读取数据并创建RDD
    rdd = sc.textFile('./data/california_housing/cal_housing.data')

    # 读取数据每个属性的定义并创建RDD
    header = sc.textFile('./data/california_housing/cal_housing.domain')
    rdd = rdd.map(lambda line: line.split(","))

    df = rdd.map(lambda line: Row(longitude=line[0],
                                  latitude=line[1],
                                  housingMedianAge=line[2],
                                  totalRooms=line[3],
                                  totalBedRooms=line[4],
                                  population=line[5],
                                  households=line[6],
                                  medianIncome=line[7],
                                  medianHouseValue=line[8])).toDF()
    df.show()
    columns = ['households', 'housingMedianAge', 'latitude', 'longitude', 'medianHouseValue', 'medianIncome',
               'population', 'totalBedRooms', 'totalRooms']

    df = convert_column(df, columns, FloatType())
    # 房屋年龄统计排序
    df.groupBy("housingMedianAge").count().sort("housingMedianAge", ascending=False).show()

    # 数据的预处理
    # 房价的值普遍都很大，我们可以把它调整成相对较小的数字；
    # 有的属性没什么意义，比如所有房子的总房间数和总卧室数，我们更加关心的是平均房间数；
    # 在我们想要构建的线性模型中，房价是结果，其他属性是输入参数。所以我们需要把它们分离处理；
    # 有的属性最小值和最大值范围很大，我们可以把它们标准化处理。

    # 所有房价都除以100000
    df = df.withColumn("medianHouseValue", col("medianHouseValue") / 100000)
    # 每个家庭的平均房间数：roomsPerHousehold
    # 每个家庭的平均人数：populationPerHousehold
    # 卧室在总房间的占比：bedroomsPerRoom
    df = df.withColumn("roomsPerHousehold", col("totalRooms") / col("households")) \
        .withColumn("populationPerHousehold", col("population") / col("households")) \
        .withColumn("bedroomsPerRoom", col("totalBedRooms") / col("totalRooms"))
    # 只选取重要的项
    df = df.select("medianHouseValue",
                   "totalBedRooms",
                   "population",
                   "households",
                   "medianIncome",
                   "roomsPerHousehold",
                   "populationPerHousehold",
                   "bedroomsPerRoom")
    # 数据分割,房价和其他
    input_data = df.rdd.map(lambda x: (x[0], DenseVector(x[1:])))
    df = spark.createDataFrame(input_data, ["label", "features"])
    df.show()
    # 数据标准化,可以用来训练模型
    standardScaler = StandardScaler(inputCol="features", outputCol="features_scaled")
    scaler = standardScaler.fit(df)
    scaled_df = scaler.transform(df)
    scaled_df.show()

    # 我们需要把数据集分为训练集和测试集，训练集用来训练模型，测试集用来评估模型的正确性。80%用来训练
    train_data, test_data = scaled_df.randomSplit([.8, .2], seed=123)

    # 构建线性回归模型
    lr = LinearRegression(featuresCol='features_scaled', labelCol="label", maxIter=10, regParam=0.3,
                          elasticNetParam=0.8)
    linearModel = lr.fit(train_data)

    # 现在有了模型，我们终于可以用linearModel的transform()函数来预测测试集中的房价，并与真实情况进行对比
    predicted = linearModel.transform(test_data)
    predictions = predicted.select("prediction").rdd.map(lambda x: x[0])
    labels = predicted.select("label").rdd.map(lambda x: x[0])
    predictionAndLabel = predictions.zip(labels).collect()

    # 我们用RDD的zip()函数把预测值和真实值放在一起，这样可以方便地进行比较。比如让我们看一下前两个对比结果
    print(predictionAndLabel)
