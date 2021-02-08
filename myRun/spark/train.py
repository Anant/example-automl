import findspark
import random
import pyspark
from sklearn import datasets
from pyspark.sql import SparkSession
import numpy
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import os
import warnings
import sys
from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print('Running')
    warnings.filterwarnings("ignore")
    findspark.init('/workspace/automl/Spark/spark-3.0.1-bin-hadoop2.7/')

    print('Setting up Spark Context/Session')
    sc = pyspark.SparkContext(appName="Iris")
    spark = SparkSession(sc)

    print('Loading Dataset')
    iris = datasets.load_iris()

    print('Getting Data')
    data = numpy.concatenate((iris["data"].tolist(), numpy.reshape(iris['target'], (len(iris['target']),-1))), 1)
    #print(data)
    irisData = sc.parallelize(data.tolist())
    irisDF = irisData.toDF(iris['feature_names']+['Species'])
    #irisDF.show()

    assembler = VectorAssembler( inputCols=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], outputCol='features')
    trainingData = assembler.transform(irisDF)
    splits = trainingData.randomSplit([0.8, 0.2], 124)
    train = splits[0]
    test = splits[1]

    num_trees = alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 10

    with mlflow.start_run():
        rf = RandomForestClassifier(labelCol="Species", featuresCol="features", numTrees=num_trees)
        model = rf.fit(train)

        predictions = model.transform(test)

        evaluator = MulticlassClassificationEvaluator(labelCol="Species", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)

        print("Random Forest Classifier Model (num_trees=%d):" % (num_trees))
        print("  Accuracy: %s %%" % (str(accuracy*100)))

        mlflow.log_param("num_trees", int(num_trees))
        mlflow.log_metric("accuracy", accuracy)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

