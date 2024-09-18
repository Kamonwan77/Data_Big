# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Create SparkSession
spark = SparkSession.builder.appName("FBLiveTH").getOrCreate()

# Load data file into DataFrame (adjust the path to your file)
data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# Use StringIndexer to create indexes for 'status_type' and 'status_published'
status_type_indexer = StringIndexer(inputCol="status_type", outputCol="status_type_ind")
status_published_indexer = StringIndexer(inputCol="status_published", outputCol="status_published_ind")

# Fit and transform the data with StringIndexer
indexed_data = status_type_indexer.fit(data).transform(data)
indexed_data = status_published_indexer.fit(indexed_data).transform(indexed_data)

# Use VectorAssembler to create features vector
assembler = VectorAssembler(inputCols=["status_type_ind", "status_published_ind"], outputCol="features")

# Create LogisticRegression model
log_reg = LogisticRegression(labelCol="status_type_ind", featuresCol="features")
log_reg.setMaxIter(10).setRegParam(0.01).setElasticNetParam(0.5)

# Create a pipeline with the stages: assembler and logistic regression
pipeline = Pipeline(stages=[assembler, log_reg])

# Split the data into train and test datasets
train_data, test_data = indexed_data.randomSplit([0.8, 0.2])

# Fit the model using train data
pipeline_model = pipeline.fit(train_data)

# Use the model to make predictions on the test data
predictions = pipeline_model.transform(test_data)

# Show 5 rows of the predictions DataFrame
predictions.select("features", "status_type_ind", "prediction").show(5)

# Create an evaluator for the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="status_type_ind", predictionCol="prediction"
)

# Evaluate the model on test data
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

# Show evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Measure: {f1}")
