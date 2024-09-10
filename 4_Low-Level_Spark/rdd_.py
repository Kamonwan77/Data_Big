from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
rdd = spark.sparkContext.parallelize(alphabet, 4)
print("Number of partitions: " + str(rdd.getNumPartitions()))

rdd2 = spark.sparkContext.textFile('C:\\Users\\gamon\\Documents\\Data\\Data_Big\\4-Low-Level_Spark\\fb_live_thailand.csv', 4)
print("Number of partitions: " + str(rdd2.getNumPartitions()))

rdd3 = spark.sparkContext.textFile('fb_live_thailand.csv',)
print("Number of partitions: " + str(rdd3.getNumPartitions()))