# Create SparkSession
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, desc, col
from graphframes import GraphFrame
import random

# Function to generate random colors
def generate_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

# Initialize Spark session with GraphFrame package
spark = SparkSession.builder \
    .appName("GraphAnalytics") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.1-spark3.0-s_2.12") \
    .getOrCreate()

# Read data from CSV file
file_path = "C:\\Users\\ADMIN\\Documents\\Data_Big\\13-powerBi-rated\\airline_routes.csv"

try:
    airline_routes_df = spark.read.csv(file_path, header=True, inferSchema=True)
    # Show the DataFrame
    airline_routes_df.show()
except Exception as e:
    print(f"Error reading CSV file: {e}")

# Create the vertex DataFrame (vertices) using withColumnRenamed() to set 'source_airport' as 'id'
vertices = airline_routes_df.select("source_airport").withColumnRenamed("source_airport", "id")

# Add a color column for each node (vertex) using random colors
vertices = vertices.withColumn("node_color", lit(generate_random_color()))

# Create the edge DataFrame (edges) using withColumnRenamed() to set 'source_airport' as 'src' and 'destination_airport' as 'dst'
edges = airline_routes_df.select("source_airport", "destination_airport") \
    .withColumnRenamed("source_airport", "src") \
    .withColumnRenamed("destination_airport", "dst")

# Create the GraphFrame
try:
    g = GraphFrame(vertices, edges)
    # Show vertices and edges
    g.vertices.show()
    g.edges.show()
except Exception as e:
    print(f"Error creating GraphFrame: {e}")

# Group the edges by 'src' and 'dst', filter count > 5, and order by count in descending order
grouped_edges = g.edges.groupBy("src", "dst").count().filter("count > 5").orderBy(desc("count"))

# Add color columns for source and destination
grouped_edges = grouped_edges.withColumn("source_color", lit("#3358FF")) \
                             .withColumn("destination_color", lit("#FF3F33"))

# Show the grouped data with colors
grouped_edges.show()

# Write the grouped data into a CSV file with overwrite mode and header
output_path = "C:\\Users\\ADMIN\\Documents\\Data_Big\\13-powerBi-rated\\grouped_edges_output.csv"

try:
    grouped_edges.write.csv(output_path, mode="overwrite", header=True)
    print(f"Data written successfully to {output_path}")
except Exception as e:
    print(f"Error writing CSV file: {e}")
