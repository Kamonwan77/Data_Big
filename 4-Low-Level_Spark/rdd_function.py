from pyspark.sql import SparkSession


spark  = SparkSession \
    .builder \
    .getOrCreate()

rdd = spark.sparkContext.textFile('fb_live_thailand.csv', 5)
print("Number of partitions: " + str(rdd.getNumPartitions()))

count_distinct = rdd.distinct().count()
print("Number of  distinct records: ", count_distinct)

filter_rdd = rdd.filter(lambda x: x.split(','[1] == 'link'.collect))
print(filter_rdd)

with tempfile.TemporaryDictionary() as data:
    folder = "textfile"
    rdd.saveAsTextFile(folder)