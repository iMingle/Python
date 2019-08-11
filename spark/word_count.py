"""Word Count

"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
from pyspark.sql.functions import regexp_replace
from pyspark.sql.functions import split


if __name__ == "__main__":
    # 初始化SparkSession程序入口
    spark = SparkSession.builder.appName("WordCount").getOrCreate()
    # 读入文档
    lines = spark.read.text("./data/shakespeare.txt")
    # 针对df特定的计算格式
    wordCounts = lines.select(explode(split(regexp_replace(lines.value, "\"", ""), "[^a-zA-Z0-9_]"))
                              .alias("word")).groupBy("word").count()
    # 展示
    wordCounts.show()
    # 关闭spark
    spark.stop()
