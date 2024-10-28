# 1. Import libraries
from pyspark.sql import SparkSession  # นำเข้า SparkSession เพื่อสร้าง session สำหรับการประมวลผลข้อมูลใน PySpark
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder  # นำเข้า StringIndexer, VectorAssembler, และ OneHotEncoder สำหรับการแปลงข้อมูลฟีเจอร์ต่างๆ
from pyspark.ml.regression import DecisionTreeRegressor  # นำเข้า DecisionTreeRegressor สำหรับสร้างโมเดลการถดถอยเชิงตัดสินใจ (Decision Tree Regression)
from pyspark.ml.evaluation import RegressionEvaluator  # นำเข้า RegressionEvaluator สำหรับประเมินโมเดลการถดถอย
from pyspark.ml import Pipeline  # นำเข้า Pipeline เพื่อจัดการลำดับขั้นตอนการแปลงข้อมูลและการฝึกโมเดล

# 2. สร้าง SparkSession สำหรับการทำงานใน PySpark
spark = SparkSession.builder \
    .appName("DecisionTreeRegressionExample") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .getOrCreate()

# 3. อ่านข้อมูลจากไฟล์ CSV ที่ชื่อ 'fb_live_thailand.csv' และกำหนดให้ Spark วิเคราะห์ชนิดข้อมูลโดยอัตโนมัติ
data = spark.read.csv("fb_live_thailand.csv", header=True, inferSchema=True)

# 4. แสดงโครงสร้างของ DataFrame และข้อมูลบางส่วนเพื่อตรวจสอบ
data.printSchema()  # แสดงโครงสร้าง
data.show(5)  # แสดงข้อมูล 5 แถวแรก

# 5. แปลงคอลัมน์ 'num_reactions' ให้เป็นตัวเลขดัชนีที่สามารถใช้งานกับโมเดลได้ โดยสร้างคอลัมน์ใหม่ 'num_reactions_ind'
indexer_reactions = StringIndexer(inputCol="num_reactions", outputCol="num_reactions_ind")

# 6. แปลงคอลัมน์ 'num_loves' ให้เป็นตัวเลขดัชนีที่สามารถใช้งานกับโมเดลได้ โดยสร้างคอลัมน์ใหม่ 'num_loves_ind'
indexer_loves = StringIndexer(inputCol="num_loves", outputCol="num_loves_ind")

# 7. แปลงคอลัมน์ดัชนี 'num_reactions_ind' ให้เป็นเวกเตอร์แบบ One-Hot โดยสร้างคอลัมน์ใหม่ 'num_reactions_vec'
encoder_reactions = OneHotEncoder(inputCols=["num_reactions_ind"], outputCols=["num_reactions_vec"])

# 8. แปลงคอลัมน์ดัชนี 'num_loves_ind' ให้เป็นเวกเตอร์แบบ One-Hot โดยสร้างคอลัมน์ใหม่ 'num_loves_vec'
encoder_loves = OneHotEncoder(inputCols=["num_loves_ind"], outputCols=["num_loves_vec"])

# 9. รวมฟีเจอร์ 'num_reactions_vec' และ 'num_loves_vec' เข้าไว้ด้วยกันเป็นคอลัมน์ 'features'
assembler = VectorAssembler(inputCols=["num_reactions_vec", "num_loves_vec"], outputCol="features")

# 10. สร้าง Pipeline ที่ประกอบไปด้วยขั้นตอนต่างๆ ตั้งแต่การแปลง StringIndexer, การเข้ารหัส OneHotEncoder และการรวมฟีเจอร์ด้วย VectorAssembler
pipeline = Pipeline(stages=[indexer_reactions, indexer_loves, encoder_reactions, encoder_loves, assembler])

# 11. ฝึก Pipeline กับข้อมูลที่อ่านเข้ามาเพื่อให้ได้ pipeline model
pipeline_model = pipeline.fit(data)

# 12. ใช้ pipeline model ในการแปลงข้อมูล ให้ได้ข้อมูลที่พร้อมสำหรับการสร้างโมเดลการทำนาย
transformed_data = pipeline_model.transform(data)

# 13. แบ่งข้อมูลออกเป็นชุดฝึก (80%) และชุดทดสอบ (20%)
train_data, test_data = transformed_data.randomSplit([0.8, 0.2])

# 14. สร้างโมเดลการถดถอยเชิงตัดสินใจ (Decision Tree Regression) โดยกำหนดค่าป้ายเป็น 'num_loves_ind' และใช้ฟีเจอร์ 'features'
dt = DecisionTreeRegressor(labelCol="num_loves_ind", featuresCol="features")

# 15. ฝึกโมเดลการถดถอยเชิงตัดสินใจด้วยข้อมูลชุดฝึก
dt_model = dt.fit(train_data)

# 16. ใช้โมเดลที่ฝึกเสร็จแล้วในการทำนายข้อมูลจากชุดทดสอบ
predictions = dt_model.transform(test_data)

# 17. สร้างตัวประเมินผลลัพธ์ของโมเดล โดยใช้ค่าป้ายเป็น 'num_loves_ind' และค่าที่ทำนายเป็น 'prediction'
evaluator = RegressionEvaluator(labelCol="num_loves_ind", predictionCol="prediction")

# 18. คำนวณค่า R-Squared (R2) เพื่อประเมินความแม่นยำของโมเดล
r2 = evaluator.setMetricName("r2").evaluate(predictions)
print(f"R2 score: {r2}")  # แสดงค่า R2 ที่คำนวณได้

# 19. ปิด SparkSession เพื่อคืนหน่วยความจำและทรัพยากร
spark.stop()
