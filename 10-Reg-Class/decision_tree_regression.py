from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline

# Create SparkSession
spark = SparkSession.builder \
    .appName("Decision Tree Regression Analysis") \
    .getOrCreate()

# Load data file into DataFrame
data = spark.read.csv('C:\\Users\\gamon\\Documents\\Data\\Data_Big\\10-Reg-Class\\fb_live_thailand.csv', header=True, inferSchema=True)
data.show()

# Use StringIndexer to create indexes for 'num_reactions' and 'num_loves'
indexer_reactions = StringIndexer(inputCol='num_reactions', outputCol='num_reactions_ind')
indexer_loves = StringIndexer(inputCol='num_loves', outputCol='num_loves_ind')

# Fit and transform the data with StringIndexer
data_indexed = indexer_reactions.fit(data).transform(data)
data_indexed = indexer_loves.fit(data_indexed).transform(data_indexed)

# Use OneHotEncoder to create Boolean flags
encoder = OneHotEncoder(
    inputCols=['num_reactions_ind', 'num_loves_ind'],
    outputCols=['num_reactions_vec', 'num_loves_vec']
)

# Use VectorAssembler to create vector of encoded columns
assembler = VectorAssembler(
    inputCols=['num_reactions_vec', 'num_loves_vec'],
    outputCol='features'
)

# Create a pipeline with stages: indexer, encoder, and assembler
pipeline = Pipeline(stages=[indexer_reactions, indexer_loves, encoder, assembler])

# Fit the DataFrame into the pipeline to create the pipeline model
pipeline_model = pipeline.fit(data_indexed)

# Use the pipeline model to transform the DataFrame data resulting in another DataFrame
data_transformed = pipeline_model.transform(data_indexed)

# Create train and test datasets using randomSplit
train_data, test_data = data_transformed.randomSplit([0.8, 0.2], seed=42)

# Create Decision Tree Regressor model
decision_tree_regressor = DecisionTreeRegressor(
    labelCol='num_loves_ind',  # Label column for the regression
    featuresCol='features'    # Features column for the regression
)

# Add Decision Tree Regressor to the pipeline
pipeline_with_regressor = Pipeline(stages=[indexer_reactions, indexer_loves, encoder, assembler, decision_tree_regressor])

# Fit the train data into the created pipeline to create the model
pipeline_model_with_regressor = pipeline_with_regressor.fit(train_data)

# Use the created model to transform the test data resulting in predictions
predictions = pipeline_model_with_regressor.transform(test_data)

# Show 5 rows of the predictions DataFrame
predictions.select('num_loves_ind', 'features', 'prediction').show(5)

# Create RegressionEvaluator
evaluator = RegressionEvaluator(
    labelCol='num_loves_ind',  # Label column for the regression
    predictionCol='prediction'  # Prediction column
)

# Calculate and show Mean Squared Error (MSE)
mse = evaluator.setMetricName("mse").evaluate(predictions)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Calculate and show R2
r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2: {r2:.4f}")
