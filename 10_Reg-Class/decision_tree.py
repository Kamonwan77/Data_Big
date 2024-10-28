# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Create SparkSession
spark = SparkSession.builder \
    .appName("Decision Tree Analysis") \
    .getOrCreate()

# Load data file into DataFrame
data = spark.read.csv('fb_live_thailand.csv', header=True, inferSchema=True)
data.show()

# Use StringIndexer to create indexes for 'status_type' and 'status_published'
indexer_type = StringIndexer(inputCol='status_type', outputCol='status_type_ind')
indexer_published = StringIndexer(inputCol='status_published', outputCol='status_published_ind')

# Fit and transform the data with StringIndexer
data_indexed = indexer_type.fit(data).transform(data)
data_indexed = indexer_published.fit(data_indexed).transform(data_indexed)
data_indexed.show()

# Use OneHotEncoder to create Boolean flags for indexed columns
encoder = OneHotEncoder(
    inputCols=['status_type_ind', 'status_published_ind'], 
    outputCols=['status_type_vec', 'status_published_vec']
)

# Transform the data with OneHotEncoder
data_encoded = encoder.fit(data_indexed).transform(data_indexed)
data_encoded.show()

# Use VectorAssembler to create a vector of encoded columns resulting in the 'features' column
assembler = VectorAssembler(
    inputCols=['status_type_vec', 'status_published_vec'],  
    outputCol='features'
)

# Create pipeline with stages: indexers, encoder, and assembler
pipeline = Pipeline(stages=[indexer_type, indexer_published, encoder, assembler])
pipeline_model = pipeline.fit(data)

# Transform the DataFrame with the pipeline model
data_transformed = pipeline_model.transform(data)
data_transformed.select('status_type', 'status_published', 'status_type_ind', 
                        'status_published_ind', 'status_type_vec', 
                        'status_published_vec', 'features').show(5)

# Create train and test datasets using randomSplit
train_data, test_data = data_transformed.randomSplit([0.8, 0.2], seed=42)

# Create Decision Tree Classifier
decision_tree = DecisionTreeClassifier(
    labelCol='status_type_ind',  
    featuresCol='features'
)

# Fit the Decision Tree model to the training data
decision_tree_model = decision_tree.fit(train_data)

# Use the created model to transform the test data resulting in predictions DataFrame
predictions = decision_tree_model.transform(test_data)
predictions.select('status_type_ind', 'features', 'prediction', 'probability').show(5)

# Use MulticlassClassificationEvaluator to create the evaluator
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

# Calculate and show Test Error
test_error = 1.0 - accuracy
print(f"Test Error: {test_error:.4f}")

# Stop SparkSession
spark.stop()
