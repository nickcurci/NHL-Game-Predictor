# Databricks notebook source
# MAGIC %md
# MAGIC # Group 3 NHL Data Analytics Project

# COMMAND ----------

# MAGIC %md
# MAGIC # Predicting event based on puck location and rink side

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load the dataset and select the appropriate fields

# COMMAND ----------

plays = spark.read.format('csv').options(header='true', inferSchema='true').load('/FileStore/tables/team_3/nhl/playss.csv').dropna()
plays_side = plays.select('x','y','event','rink_side')
display(plays_side)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create rink side as an integer

# COMMAND ----------

from pyspark.sql.functions import *
plays_selected=plays_side.withColumn('s', when(col('rink_side') == ('left'), 0).when(col('rink_side') == ('right'), 1))
display(plays_selected)

# COMMAND ----------

# MAGIC %md
# MAGIC ### drop nulls

# COMMAND ----------

playsDF=plays_selected.dropna()
playsDF.show(2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Count and show the number of events

# COMMAND ----------

playsDF.groupBy('event').count().show()

# COMMAND ----------

playsDF.count()

# COMMAND ----------

# MAGIC %md
# MAGIC ### show the schema

# COMMAND ----------


playsDF.cache()
playsDF.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### vector assembly, create a features column with X, Y, and the integer for rink side

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vectorAssembler = VectorAssembler(inputCols = ['x','y','s'], outputCol = 'features')

playside_df = vectorAssembler.transform(playsDF)

playside_df.select(['features', 'event']).show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### create a label column as event

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

labelIndex = StringIndexer().setInputCol('event').setOutputCol('label')

# COMMAND ----------

# MAGIC %md
# MAGIC ### fit the dataset and drop any nulls (again, just in case)

# COMMAND ----------

label_DF = labelIndex.fit(playside_df).transform(playside_df).dropna()

# COMMAND ----------

display(label_DF)

# COMMAND ----------

# MAGIC %md
# MAGIC ### split the new data into testing and training datasets

# COMMAND ----------

(atrain, atest) = label_DF.randomSplit([0.7, 0.3], seed=100)

# COMMAND ----------

# MAGIC %md
# MAGIC ### run logistic regression

# COMMAND ----------

from pyspark.ml.classification import *

log = LogisticRegression(featuresCol = 'features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
log_model = log.fit(atrain)

# COMMAND ----------

# MAGIC %md
# MAGIC ### show the prediction

# COMMAND ----------

logPrediction=log_model.transform(atest)
logPrediction.select("label", "prediction").show(10, False)

# COMMAND ----------

# MAGIC %md
# MAGIC ### calculate the prediction accuracy

# COMMAND ----------

print("prediction accuracy is: ", logPrediction.where("prediction==label").count()/logPrediction.count())

# COMMAND ----------

# MAGIC %md
# MAGIC # The prediction accuracy is very low at 23.2%
# MAGIC ## Meaning puck location and rink side are not indiciative of the event that will happen

# COMMAND ----------

# MAGIC %md
# MAGIC # Prediction accuracy increased from 23.07% to 23.2% when rink side was added in

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross Validation

# COMMAND ----------

# instantiate a logistic Regression model
from pyspark.ml.classification import LogisticRegression

lr= LogisticRegression()

 # Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.1, 2])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [5, 10])
             .build())

#define evaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator()\
  .setMetricName("areaUnderROC")\
  .setRawPredictionCol("prediction")\
  .setLabelCol("label")

# Create 2-fold CrossValidator
cv_lr = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=2)

# Run cross validations. 
# this will likely take a fair amount of time because of the amount of models that we're creating and testing. 
# It takes 3 minutes to run this model.
cv_lrModel = cv_lr.fit(atrain)

# COMMAND ----------

# Use test set to measure the accuracy of our model on new data
cv_lrPrediction = cv_lrModel.transform(atest)

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(cv_lrPrediction)

# COMMAND ----------

# Calculate accuracy

print("prediction accuracy is: ", cv_lrPrediction.where("prediction==label").count()/cv_lrPrediction.count())

# COMMAND ----------

# MAGIC %md
# MAGIC # Cross Validation did NOT improve the model at all.
