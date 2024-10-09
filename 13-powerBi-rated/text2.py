from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Create SparkSession
spark = SparkSession.builder.appName("TextAnalytics").getOrCreate()

# Read data from file (replace 'reviews_rated.csv' with your actual file)
data = spark.read.csv("C:\\Users\\ADMIN\\Documents\\Data_Big\\13-powerBi-rated\\reviews_rated.csv", header=True, inferSchema=True)

# Select Review Text and Rating columns, trim Review Text, and cast Rating to IntegerType
data = data.select(data["Review Text"].alias("review_text"), data["Rating"].cast(IntegerType()).alias("rating"))
data = data.na.drop()  # Drop rows with missing values
data.show(5)

# Tokenizer: Split text into words
tokenizer = Tokenizer(inputCol="review_text", outputCol="words")

# StopWordsRemover: Remove common words
stopword_remover = StopWordsRemover(inputCol="words", outputCol="meaningful_words")

# HashingTF: Convert words to features using Term Frequency
hashing_tf = HashingTF(inputCol="meaningful_words", outputCol="raw_features")

# IDF: Inverse Document Frequency to downweight common words
idf = IDF(inputCol="raw_features", outputCol="features")

# Create a Pipeline to apply the transformations sequentially
pipeline = Pipeline(stages=[tokenizer, stopword_remover, hashing_tf, idf])

# Split the data into training and testing sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=1234)

# Fit the pipeline to training data
pipeline_model = pipeline.fit(train_data)

# Transform training and testing data
train_transformed = pipeline_model.transform(train_data)
test_transformed = pipeline_model.transform(test_data)

# Show the transformed training data
train_transformed.select("meaningful_words", "features", "rating").show(5)

# Logistic Regression Model with parameter tuning
log_reg = LogisticRegression(labelCol="rating", featuresCol="features", maxIter=200, regParam=0.09)

# Fit the model to the transformed train dataset
log_reg_model = log_reg.fit(train_transformed)

# Predict on the test dataset
predictions = log_reg_model.transform(test_transformed)

# Show the predictions
predictions.select("meaningful_words", "rating", "prediction").show(5)

# Evaluate the model using MulticlassClassificationEvaluator
evaluator = MulticlassClassificationEvaluator(labelCol="rating", predictionCol="prediction", metricName="accuracy")

# Calculate accuracy
accuracy = evaluator.evaluate(predictions)
print(f"Accuracy: {accuracy}")
