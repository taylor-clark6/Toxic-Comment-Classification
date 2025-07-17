from pyspark.sql import SparkSession
from pyspark.sql.functions import col

#initialize Spark session
spark = SparkSession.builder.appName("TextToSparseVector").getOrCreate()

#read CSV
df = spark.read.csv("train.csv", header=True, inferSchema=True).dropna(subset=["comment_text"])

# Combine multiple toxic labels into a single binary label column
toxic_columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
df = df.withColumn("label", (sum([col(c) for c in toxic_columns]) > 0).cast("int"))

#tokenize and transform comment_text to sparse vectors
from pyspark.ml.feature import Tokenizer, HashingTF, IDF

tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
words_data = tokenizer.transform(df)

hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
featurized_data = hashing_tf.transform(words_data)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_data)
rescaled_data = idf_model.transform(featurized_data)

#remove rows with null labels BEFORE modeling or output
df_labeled = rescaled_data.filter(rescaled_data["label"].isNotNull())

#preview output
df_labeled.select("comment_text", "label", "features").show(10, truncate=False)
