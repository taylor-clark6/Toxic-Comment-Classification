#!/bin/bash
source ../../../env.sh
echo "Using SPARK_MASTER=$SPARK_MASTER"
/usr/local/spark/bin/spark-submit --master=spark://$SPARK_MASTER:7077 --conf spark.default.parallelism=4 ./spark_text_classification.py sample_submission.csv test.csv test_labels.csv train.csv
