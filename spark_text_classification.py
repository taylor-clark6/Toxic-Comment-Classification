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

"""
#tokenize text
from pyspark.ml.feature import Tokenizer

tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
words_data = tokenizer.transform(df)

#convert words to term frequencies (sparse vectors)
from pyspark.ml.feature import HashingTF

hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=10000)
featurized_data = hashing_tf.transform(words_data)

#apply TF-IDF
from pyspark.ml.feature import IDF

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_data)
rescaled_data = idf_model.transform(featurized_data)

#preview output
rescaled_data.select("comment_text", "features").show(truncate=False)

#stop the spark session
spark.stop()
"""

'''
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression

if __name__ == "__main__":
    # Build a spark context
    hc = (SparkSession.builder
                      .appName('Toxic Comment Classification')
                      .master("spark://10.128.0.2:7077")
                      .enableHiveSupport()
                      .config("spark.executor.memory", "512m") #was 4G
                      .config("spark.driver.memory","1G") #was 18G
                      .config("spark.executor.cores","1") #was 7
                      .config("spark.python.worker.memory","512m") #was 4G
                      .config("spark.driver.maxResultSize","0")
                      .config("spark.sql.crossJoin.enabled", "true")
                      .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
                      .config("spark.default.parallelism","2")
                      .getOrCreate())

    hc.sparkContext.setLogLevel('INFO')

    hc.conf.set("spark.sql.shuffle.partitions", "2")  # default is 200

    #hc.version

    def to_spark_df(fin):
        """
        Parse a filepath to a spark dataframe using the pandas api.

        Parameters
        ----------
        fin : str
            The path to the file on the local filesystem that contains the csv data.

        Returns
        -------
        df : pyspark.sql.dataframe.DataFrame
            A spark DataFrame containing the parsed csv data.
        """
        df = hc.read.option("header", "true").option("inferSchema", "true").csv(fin)
        df = df.fillna("")  # fill missing values with empty string
        return df

    # Load the train-test sets
    train = to_spark_df("train.csv")
    test = to_spark_df("test.csv")

    out_cols = [i for i in train.columns if i not in ["id", "comment_text"]]

    # Cast label columns to DoubleType
    from pyspark.sql.types import DoubleType
    for col in out_cols:
        train = train.withColumn(col, train[col].cast(DoubleType()))

    # Sadly the output is not as  pretty as the pandas.head() function
    #train.show(5)

    # View some toxic comments
    #train.filter(F.col('toxic') == 1).show(5)

    # Basic sentence tokenizer
    tokenizer = Tokenizer(inputCol="comment_text", outputCol="words")
    wordsData = tokenizer.transform(train)

    # Count the words in a document
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=2048) #2048 features
    tf = hashingTF.transform(wordsData)

    #tf.select('rawFeatures').take(2)

    # Build the idf model and transform the original token frequencies into their tf-idf counterparts
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(tf)
    tfidf = idfModel.transform(tf)

    # Drop rows with nulls in any label column before training
    tfidf = tfidf.dropna(subset=out_cols)

    # Persist to avoid recomputation during repeated training
    tfidf = tfidf.persist()

    #tfidf.select("features").first()

    #REG = 0.1

    #lr = LogisticRegression(featuresCol="features", labelCol='toxic', regParam=REG)

    #tfidf.show(5)

    #lrModel = lr.fit(tfidf.limit(5000))

    #res_train = lrModel.transform(tfidf)

    #res_train.select("id", "toxic", "probability", "prediction").show(20)

    #res_train.show(5)

    #extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())

    #(res_train.withColumn("proba", extract_prob("probability"))
    # .select("proba", "prediction")
    # .show())

    test_tokens = tokenizer.transform(test)
    test_tf = hashingTF.transform(test_tokens)
    test_tfidf = idfModel.transform(test_tf).persist()

    #test_res = test.select('id')
    #test_res.head()

    from pyspark.ml.functions import vector_to_array
    from functools import reduce

    #extract_prob = F.udf(lambda x: float(x[1]), T.FloatType())
    predictions = [] #collect predictions in a list and merge them once at the end

    for col in out_cols:
        print(f"Training and predicting for: {col}")
        lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=0.1)
        lrModel = lr.fit(tfidf.limit(5000))
    
        res = lrModel.transform(test_tfidf)
        prob = vector_to_array("probability")[1].alias(col)
        predictions.append(res.select("id", prob))

    # Repartition each prediction DataFrame by 'id' before joining
    predictions = [df.repartition("id") for df in predictions]

    # Merge predictions on 'id'
    test_res = reduce(lambda df1, df2: df1.join(df2, on="id"), predictions)

    """
    test_probs = []
    for col in out_cols:
        print(col)
        lr = LogisticRegression(featuresCol="features", labelCol=col, regParam=REG)
        print("...fitting")
        lrModel = lr.fit(tfidf.dropna(subset=[col]).limit(5000))
        print("...predicting")
        res = lrModel.transform(test_tfidf)
        print("...appending result")
        test_res = test_res.join(res.select('id', 'probability'), on="id")
        print("...extracting probability")
        test_res = test_res.withColumn(col, extract_prob('probability')).drop("probability")
        #test_res.show(5)
    """
    #test_res.show(5)

    # Write to disk
    test_res.coalesce(1).write.csv('./results/spark_lr.csv', mode='overwrite', header=True)

    # Cleanup
    tfidf.unpersist()
    test_tfidf.unpersist()

    hc.stop()
'''
