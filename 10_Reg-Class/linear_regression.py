from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

import seaborn as sns
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, desc
from pyspark.sql.types import IntegerType

# Create SparkSession
spark = SparkSession.builder \
    .appName("Linear Regression Analysis") \
    .getOrCreate()

# Load data file into DataFrame
data = spark.read.csv('fb_live_thailand.csv', header=True, inferSchema=True)
data.show()

# Use VectorAssembler to create vector of 'num_reactions' and 'num_loves'
assembler = VectorAssembler(
    inputCols=['num_reactions', 'num_loves'],
    outputCol='features'
)

# Transform the data with VectorAssembler
data_assembled = assembler.transform(data)
data_assembled.show()

# Create Linear Regression model
linear_regression = LinearRegression(
    labelCol='num_loves',  # Label column for the regression
    featuresCol='features',    # Features column for the regression
    maxIter=10,                # Set maximum number of iterations (adjust as needed)
    regParam=0.3,              # Set regularization parameter (0...1, adjust as needed)
    elasticNetParam=0.8        # Set ElasticNet mixing parameter (0...1, adjust as needed)
)

# Create a pipeline with the stages including only the linear regression model
pipeline = Pipeline(stages=[linear_regression])

# Create train and test datasets using randomSplit
train_data, test_data = data_assembled.randomSplit([0.8, 0.2], seed=42)

# Fit train data into the created pipeline to create the pipeline model
pipeline_model = pipeline.fit(train_data)

# Use the created pipeline model to transform test data resulting in the predictions DataFrame
predictions = pipeline_model.transform(test_data)

# Show 5 rows of the predictions DataFrame
predictions.select('num_loves', 'features', 'prediction').show(5)

# Create RegressionEvaluator
evaluator = RegressionEvaluator(
    labelCol='num_loves',  # Label column for the regression
    predictionCol='prediction'  # Prediction column
)

# Calculate and show Mean Squared Error (MSE)
mse = evaluator.setMetricName("mse").evaluate(predictions)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Calculate and show R2
r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2: {r2:.4f}")

# Convert Spark DataFrame to Pandas DataFrame
pandas_df = predictions.select('num_loves', 'prediction').toPandas()

# Plotting with seaborn and matplotlib
plt.figure(figsize=(10, 6))
sns.scatterplot(x='num_loves', y='prediction', data=pandas_df)
plt.title('Scatter Plot of num_loves vs Prediction')
plt.xlabel('num_loves')
plt.ylabel('Prediction')
plt.show()

# Select num_loves and prediction columns
# Convert these columns to IntegerType and order by prediction column in descending order
selected_data = predictions.select(
    col('num_loves').cast(IntegerType()).alias('num_loves'),
    col('prediction').cast(IntegerType()).alias('prediction')
).orderBy(col('prediction').desc())

# Convert selected data to Pandas DataFrame
pandas_df = selected_data.toPandas()

# Plot using seaborn lmplot
plt.figure(figsize=(10, 6))
sns.lmplot(x='num_loves', y='prediction', data=pandas_df, aspect=1.5)

# Show the plot
plt.title('Linear Regression: num_loves vs Prediction')
plt.xlabel('num_loves')
plt.ylabel('Prediction')
plt.show()
