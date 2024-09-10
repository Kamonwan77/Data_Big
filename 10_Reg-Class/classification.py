# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Create SparkSession
spark = SparkSession.builder \
    .appName("FBLiveTH Analysis") \
    .getOrCreate()

# Load data file into DataFrame
data = spark.read.csv('C:\\Users\\gamon\\Documents\\Data\\Data_Big\\10-Reg-Class\\fb_live_thailand.csv', header=True, inferSchema=True)

# Show the loaded data
data.show()

# Use StringIndexer to prepare data
indexer_type = StringIndexer(inputCol='status_type', outputCol='status_type_ind')
indexer_published = StringIndexer(inputCol='status_published', outputCol='status_published_ind')

# Fit and transform the data
data_indexed = indexer_type.fit(data).transform(data)
data_indexed = indexer_published.fit(data_indexed).transform(data_indexed)

# Show the data with indexed columns
data_indexed.show()

# Use VectorAssembler to create a vector of 'status_type_ind' and 'status_published_ind'
assembler = VectorAssembler(inputCols=['status_type_ind', 'status_published_ind'], outputCol='features')
data_assembled = assembler.transform(data_indexed)

# Show the final DataFrame with the features column
data_assembled.show()

# Split the data into train and test datasets using randomSplit
train_data, test_data = data_assembled.randomSplit([0.8, 0.2], seed=42)

# Create logistic regression model
log_reg = LogisticRegression(
    labelCol='status_type_ind', 
    featuresCol='features', 
    maxIter=10,                # Set maximum number of iterations
    regParam=0.3,              # Set regularization parameter
    elasticNetParam=0.8        # Set ElasticNet mixing parameter
)

# Create a pipeline with stages including assembler and logistic regression
pipeline = Pipeline(stages=[log_reg])

# Fit the pipeline using the train data
pipeline_model = pipeline.fit(train_data)

# Use the created pipeline model to transform the test data
predictions = pipeline_model.transform(test_data)

# Show 5 rows of the predictions DataFrame
predictions.select('status_type_ind', 'status_published_ind', 'features', 'prediction', 'probability').show(5)

# Create evaluator for classification
evaluator = MulticlassClassificationEvaluator(
    labelCol='status_type_ind', 
    predictionCol='prediction'
)

# Evaluate accuracy
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
print(f"Accuracy: {accuracy:.4f}")

# Evaluate weighted precision
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
print(f"Weighted Precision: {precision:.4f}")

# Evaluate weighted recall
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
print(f"Weighted Recall: {recall:.4f}")

# Evaluate F1 measure
f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
print(f"F1 Measure: {f1:.4f}")

# Stop SparkSession
spark.stop()
