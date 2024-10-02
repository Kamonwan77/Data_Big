# Import Libraries
from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import desc

# Create SparkSession with GraphFrames
spark = SparkSession.builder \
    .appName("GraphAnalytics") \
    .config("spark.driver.memory", "4g") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
    .getOrCreate()

# Create vertices DataFrame
vertices = spark.createDataFrame([
    ("Alice", "45"),
    ("Jacob", "43"),
    ("Roy", "21"),
    ("Ryan", "49"),
    ("Emily", "24"),
    ("Sheldon", "52"),
], ["id", "age"])

# Create edges DataFrame
edges = spark.createDataFrame([
    ("Sheldon", "Alice", "Sister"),
    ("Alice", "Jacob", "Husband"),
    ("Emily", "Jacob", "Father"),
    ("Ryan", "Alice", "Friend"),
    ("Alice", "Emily", "Daughter"),
    ("Alice", "Roy", "Son"),
    ("Jacob", "Roy", "Son"),
], ["src", "dst", "relation"])

# Create GraphFrame
try:
    graph = GraphFrame(vertices, edges)
    print("GraphFrame created successfully.")
except Exception as e:
    print(f"Error creating GraphFrame: {e}")

# Show vertices and edges
print("Vertices:")
vertices.show()

print("Edges:")
edges.show()

# Group and order the nodes and edges
grouped_edges = graph.edges.groupBy("src", "dst").count().orderBy(desc("count"))
print("Grouped Edges:")
grouped_edges.show(5)

# Filter the GraphFrame
filtered_edges = graph.edges.where("src = 'Alice' OR dst = 'Jacob'").groupBy("src", "dst").count().orderBy(desc("count"))
print("Filtered Edges:")
filtered_edges.show(5)

# Create a subgraph
subgraph_query = graph.edges.where("src = 'Alice' OR dst = 'Jacob'")
subgraph = GraphFrame(graph.vertices, subgraph_query)
print("Subgraph Edges:")
subgraph.edges.show()

# Find motifs
motifs = graph.find("(a) - [ab] -> (b)")
print("Motifs:")
motifs.show()

# PageRank
rank = graph.pageRank(resetProbability=0.15, maxIter=5)
print("PageRank:")
rank.vertices.orderBy(desc("pagerank")).show(5)

# In-Degree and Out-Degree
in_degree = graph.inDegrees
print("In-Degree:")
in_degree.orderBy(desc("inDegree")).show(5)

out_degree = graph.outDegrees
print("Out-Degree:")
out_degree.orderBy(desc("outDegree")).show(5)

# Connected Components
try:
    connected_components = graph.connectedComponents()
    print("Connected Components:")
    connected_components.show()
except Exception as e:
    print(f"Error calculating connected components: {e}")

# Strongly Connected Components
scc = graph.stronglyConnectedComponents(maxIter=5)
print("Strongly Connected Components:")
scc.show()

# Breadth-First Search (BFS)
bfs_result = graph.bfs(fromExpr="id = 'Alice'", toExpr="id = 'Roy'", maxPathLength=2)
print("BFS Result from id 'Alice' to id 'Roy' with maxPathLength = 2:")
bfs_result.show()

# Stop SparkSession when done
spark.stop()
