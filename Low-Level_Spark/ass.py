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

flatmap = rdd.flatMap(lambda x: x.split(','))
pair = flatmap.map(lambda x: (x,1))
take_pair = pair.take(200)
for f in take_pair:
    if str(f[0]) == 'photo' or str(f[0]) == 'video':
        print(str(f[0]), str(f[1]))

sort_data = pair.sortByKey().collect()
for f in sort_data:
    print(str(f[0]), str(f[1]))

sort_data = pair.sortBy(lambda x:x, False, 5).collect()
for f in sort_data:
    print(str(f[0]), str(f[1]))

reduce_key = pair.reduceByKey(lambda x,y: x+y)
print(reduce_key.take(10))

# tranform
alphabet = [('a',1), ('b',2), ('c',3), ('a',1), ('b',2)]
rdd = spark.sparkContext.parallelize(alphabet, 4)
#1
def tolist(x):
    return [x]
def append(x, y):
    x.append(y)
    return x
def extend(x, y):
    x.extend(y)
    return x
combine = sorted(rdd.combineByKey(tolist, append, extend).take(10))
print(combine)

#2
zero_val = (0,0)
par_agg = lambda x,y: (x[0] + y, x[1] + 1)
allpar_agg = lambda x,y: (x[0] + y[0], x[1] + y[1])
agg = rdd.aggregateByKey(zero_val, par_agg, allpar_agg).take(10)
print(agg)

#3
from operator import add
fold = sorted(rdd.foldByKey(0, add).collect())
print(fold)

#4
group1 = sorted(rdd.groupByKey().mapValues(len).take(10))
print(group1)
group2 = sorted(rdd.groupByKey().mapValues(list).take(10))
print(group2)


# alphabet1
alphabet1 = [('a',1), ('b',2), ('c',3)]
rdd1 = spark.sparkContext.parallelize(alphabet1)

alphabet2 = [('a',1), ('b',2), ('a',1), ('b',2)]
rdd2 = spark.sparkContext.parallelize(alphabet2)
joinRDD = rdd1.join(rdd2).collect()
print(joinRDD)
left = rdd1.leftOuterJoin(rdd2).collect()
print(left)
right = rdd1.rightOuterJoin(rdd2).collect()
print(right)
count = rdd.countByKey()
print(count)
count_val = rdd.countByValue()
print(count_val)
col_asmap = rdd.collectAsMap()
print(col_asmap)
look = rdd.lookup('a')
print(look)
first = rdd.first()
print(first)
max = rdd.max()
print(max)
min = rdd.min()
print(min)


