from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc, col, lit


spark = SparkSession.builder \
    .appName("Graph Analytics Assignment") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

airline_routes_df = spark.read.csv("airline_routes.csv", header=True, inferSchema=True)

airline_routes_df.show()

vertices_df = airline_routes_df.select("source_airport").withColumnRenamed("source_airport", "id").distinct()

edges_df = airline_routes_df.select("source_airport", "destination_airport") \
    .withColumnRenamed("source_airport", "src") \
    .withColumnRenamed("destination_airport", "dst")

vertices_df.show()
edges_df.show()

graph = GraphFrame(vertices_df, edges_df)

print("Number of vertices:", graph.vertices.count())
print("Number of edges:", graph.edges.count())

grouped_edges_df = graph.edges.groupBy("src", "dst").count() \
    .filter(col("count") > 5) \
    .orderBy(desc("count")) \
    .withColumn("source_color", lit("#3358FF")) \
    .withColumn("destination_color", lit("#FF3F33"))

grouped_edges_df.show()

grouped_edges_df.write.csv("output.csv", mode="overwrite", header=True)