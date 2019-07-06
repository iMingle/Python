"""纽约出租车载客信息
用Structured Streaming对纽约市出租车的载客信息进行处理，建立一个实时流处理的pipeline，
实时输出各个区域内乘客小费的平均数来帮助司机决定要去哪里接单

数据地址:https://training.ververica.com/setup/taxiData.html

"""
import os

from pyspark.sql import SparkSession, window
from pyspark.sql.functions import expr, avg
from pyspark.sql.functions import split
from pyspark.sql.types import *


os.environ["PYSPARK_PYTHON"] = "/usr/local/bin/python3"


def parse_data_from_kafka_message(sdf, schema):
    assert sdf.isStreaming == True, "DataFrame doesn't receive streaming data"
    col = split(sdf['value'], ',')
    for idx, field in enumerate(schema):
        sdf = sdf.withColumn(field.name, col.getItem(idx).cast(field.dataType))
    return sdf.select([field.name for field in schema])


if __name__ == "__main__":
    spark = SparkSession.builder.appName("Spark Structured Streaming for taxi ride info").getOrCreate()

    # 流数据输入
    rides = spark.readStream.format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:xxxx") \
        .option("subscribe", "taxirides").option("startingOffsets", "latest").load().selectExpr("CAST(value AS STRING)")

    fares = spark.readStream.format("kafka").option("kafka.bootstrap.servers", "localhost:xxxx") \
        .option("subscribe", "taxifares").option("startingOffsets", "latest").load().selectExpr("CAST(value AS STRING")

    # 数据清洗
    ridesSchema = StructType([
        StructField("rideId", LongType()), StructField("isStart", StringType()),
        StructField("endTime", TimestampType()), StructField("startTime", TimestampType()),
        StructField("startLon", FloatType()), StructField("startLat", FloatType()),
        StructField("endLon", FloatType()), StructField("endLat", FloatType()),
        StructField("passengerCnt", ShortType()), StructField("taxiId", LongType()),
        StructField("driverId", LongType())])

    faresSchema = StructType([
        StructField("rideId", LongType()), StructField("taxiId", LongType()),
        StructField("driverId", LongType()), StructField("startTime", TimestampType()),
        StructField("paymentType", StringType()), StructField("tip", FloatType()),
        StructField("tolls", FloatType()), StructField("totalFare", FloatType())])
    rides = parse_data_from_kafka_message(rides, ridesSchema)
    fares = parse_data_from_kafka_message(fares, faresSchema)

    # 所有代表出发的事件都已被删除，因为到达事件已经包含了出发事件的所有信息，而且只有到达之后才会付费
    # 出发地点和目的地在纽约范围外的数据，也可以被删除。因为我们的目标是找出纽约市内小费较高的地点。
    # DataFrame的filter函数可以很容易地做到这些
    MIN_LON, MAX_LON, MIN_LAT, MAX_LAT = -73.7, -74.05, 41.0, 40.5
    rides = rides.filter(
        rides["startLon"].between(MIN_LON, MAX_LON) &
        rides["startLat"].between(MIN_LAT, MAX_LAT) &
        rides["endLon"].between(MIN_LON, MAX_LON) &
        rides["endLat"].between(MIN_LAT, MAX_LAT))
    rides = rides.filter(rides["isStart"] == "END")

    # stream和stream的join操作需要借助数据水印(Watermark),数据水印定义了我们可以对数据延迟的最大容忍限度
    faresWithWatermark = fares.selectExpr("rideId AS rideId_fares", "startTime", "totalFare", "tip") \
        .withWatermark("startTime", "30 minutes")

    ridesWithWatermark = rides.selectExpr("rideId", "endTime", "driverId", "taxiId", "startLon",
                                          "startLat", "endLon", "endLat").withWatermark("endTime", "30 minutes")

    joinDF = faresWithWatermark.join(ridesWithWatermark, expr("""rideId_fares = rideId AND
               endTime > startTime AND
               endTime <= startTime + interval 2 hours"""))
    # 纽约市的区域信息以及坐标可以从网上找到，这部分处理比较容易。每个接收到的数据我们都可以判定它在哪个区域内，
    # 然后对 joinDF 增加一个列“area”来代表终点的区域。现在，让我们假设area已经加到现有的DataFrame里
    tips = joinDF.groupBy(window("endTime", "30 minutes", "10 minutes"), "area").agg(avg("tip"))

    tips.writeStream.outputMode("append").format("console").option("truncate", False).start().awaitTermination()
