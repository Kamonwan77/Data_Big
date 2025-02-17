from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth
from pyspark.sql.functions import collect_list, array_distinct, explode, split, col

# Step 1: Create SparkSession
spark = SparkSession.builder.appName("FPGrowthExample").getOrCreate()

# Step 2: Read data (assuming CSV format for this example)
data = spark.read.csv("groceries_data.csv", header=True, inferSchema=True)

# Step 3: Group data based on member number
grouped_data = data.groupBy("Member_number").agg(collect_list("itemDescription").alias("Items"))

# Step 4: Add a column 'basket' with unique items
grouped_data = grouped_data.withColumn("basket", array_distinct(col("Items")))

# Step 5: Explode the Items array to separate items into rows
exploded_data = grouped_data.select("Member_number", explode("basket").alias("item"))

# Step 6: Split items where necessary (assuming '/' should be replaced with ',')
# Note: Adjust this step based on your actual data if needed
separated_data = exploded_data.withColumn("item", explode(split(col("item"), "/")))

# Step 7: Group the separated items back into lists and ensure they are unique
final_data = separated_data.groupBy("Member_number").agg(collect_list("item").alias("Items"))

# Step 8: Ensure Items are unique again
final_data = final_data.withColumn("Items", array_distinct(col("Items")))

# Step 9: Create FPGrowth model with specified parameters
minSupport = 0.1
minConfidence = 0.2
fp = FPGrowth(minSupport=minSupport, minConfidence=minConfidence, itemsCol='Items', predictionCol='prediction')

# Step 10: Fit FPGrowth model to the final data
model = fp.fit(final_data)

# Step 11: Show frequent itemsets
model.freqItemsets.show(10)

# Step 12: Filter association rules based on confidence
filtered_rules = model.associationRules.filter(col("confidence") > 0.4)

# Step 13: Show filtered rules
filtered_rules.show(truncate=False)

# Step 14: Create a new DataFrame for predictions
new_data = spark.createDataFrame(
    [
        (["vegetable juice", "frozen fruits", "packaged fruit"],),
        (["mayonnaise", "butter", "buns"],)
    ],
    ["Items"]
)

# Step 15: Transform the model with the new data for predictions
predictions = model.transform(new_data)

# Step 16: Show predictions
predictions.show(truncate=False)

# Stop Spark session
spark.stop()
