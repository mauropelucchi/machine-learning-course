/*
 * Copyright 2017 Mauro Pelucchi
 *
 * See LICENSE file for further information.
 */


//
// bin/spark-shell --driver-memory=4g
// 
// Forest Cover type Data Set
//
// Data from https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/
// 

import org.apache.spark.ml.linalg._
import org.apache.spark.ml.regression._


val data_raw = spark.read.option("inferSchema", true).option("header", false).csv("/Users/mauropelucchi/Desktop/Machine_Learning/Dataset/Covtype/covtype.data")

// check data
data_raw.count
// about 581012
data_raw.printSchema

// 54 features

// prepare data
val col_name = Seq("Elevation", "Aspect", "Slope","Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology","Horizontal_Distance_To_Roadways","Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm","Horizontal_Distance_To_Fire_Points") ++ ((0 until 4).map(i => s"Wilderness_Area_$i")) ++ ((0 until 40).map(i => s"Soil_Type_$i")) ++ Seq("Cover_Type")


val data = data_raw.toDF(col_name:_*).withColumn("Cover_Type", $"Cover_Type".cast("double"))
data.count
data.printSchema

data.show()
data.head

// Split into 80% train, 10% cross validation, 10% test
val Array(trainData, cvData, testData) = data.randomSplit(Array(0.8,0.1, 0.1))
trainData.cache()
cvData.cache()
testData.cache()





//
// first simple decision tree
// Target --> Cover_type column
//
import org.apache.spark.ml.feature.{VectorAssembler, VectorIndexer}
import org.apache.spark.ml.classification.{DecisionTreeClassifier,RandomForestClassifier, RandomForestClassificationModel}

val inputCols = trainData.columns.filter(_ != "Cover_Type")
val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featureVector")

val assembledTrainData = assembler.transform(trainData)
assembledTrainData.select("featureVector").show(truncate = false)

val assembledCvData = assembler.transform(cvData)
assembledCvData.select("featureVector").show(truncate = false)

val classifier = new DecisionTreeClassifier().setSeed(2345).setLabelCol("Cover_Type").setFeaturesCol("featureVector").setPredictionCol("prediction")

// train the model
val model = classifier.fit(assembledTrainData)

// show the tree
println(model.toDebugString)

// explore the model
model.featureImportances.toArray.zip(inputCols).sorted.reverse.take(10).foreach(println)

// make predictions
val predictions = model.transform(assembledCvData)

predictions.select("Cover_Type", "prediction", "probability").show(false)

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val evaluator = new MulticlassClassificationEvaluator().setLabelCol("Cover_Type").setPredictionCol("prediction")

val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
val f1 = evaluator.setMetricName("f1").evaluate(predictions)
println(accuracy)
println(f1)

val weightedPrecision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
println(weightedPrecision)

// create confunsion matrix
import org.apache.spark.mllib.evaluation.MulticlassMetrics
val predictionRDD = predictions.select("prediction", "Cover_Type").as[(Double,Double)].rdd
val multiclassMetrics = new MulticlassMetrics(predictionRDD)
println(multiclassMetrics.confusionMatrix)

val confusionMatrix = predictions.groupBy("Cover_Type").pivot("prediction", (1 to 7)).count().na.fill(0.0).orderBy("Cover_Type")
confusionMatrix.show()



// tuning hyperparameter


import org.apache.spark.ml.{PipelineModel, Pipeline}
val inputCols = trainData.columns.filter(_ != "Cover_Type")
val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featureVector")

val classifier = new DecisionTreeClassifier().setSeed(23456).setLabelCol("Cover_Type").setFeaturesCol("featureVector").setPredictionCol("prediction")

val pipeline = new Pipeline().setStages(Array(assembler, classifier))

// create grid from tuning decision trees
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
val paramGrid = new ParamGridBuilder().addGrid(classifier.impurity, Seq("gini", "entropy")).addGrid(classifier.maxDepth, Seq(1, 20)).addGrid(classifier.maxBins, Seq(40, 300)).addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).build()

val multiclassEval = new MulticlassClassificationEvaluator().setLabelCol("Cover_Type").setPredictionCol("prediction").setMetricName("accuracy")

val validator = new TrainValidationSplit().setSeed(12345).setEstimator(pipeline).setEvaluator(multiclassEval).setEstimatorParamMaps(paramGrid).setTrainRatio(1.0)
val validatorModel = validator.fit(trainData)

val paramsAndMetrics = validatorModel.validationMetrics.zip(validatorModel.getEstimatorParamMaps).sortBy(-_._1)

paramsAndMetrics.foreach { case (metric, params) =>
        println(metric)
        println(params)
        println()}

val bestModel = validatorModel.bestModel
println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)
println(validatorModel.validationMetrics.max)

val cvAccuracy = multiclassEval.evaluate(bestModel.transform(cvData))
println(cvAccuracy)

val testAccuracy = multiclassEval.evaluate(bestModel.transform(testData))
println(testAccuracy)

val trainAccuracy = multiclassEval.evaluate(bestModel.transform(trainData))
println(trainAccuracy)




// evaluate categorical encoding
// from https://github.com/sryza/aas/blob/master/ch04-rdf/src/main/scala/com/cloudera/datascience/rdf/RunRDF.scala
import org.apache.spark.sql.{DataFrame, SparkSession}
 def unencodeOneHot(data: DataFrame): DataFrame = {
    val wildernessCols = (0 until 4).map(i => s"Wilderness_Area_$i").toArray

    val wildernessAssembler = new VectorAssembler().
      setInputCols(wildernessCols).
      setOutputCol("wilderness")

    val unhotUDF = udf((vec: Vector) => vec.toArray.indexOf(1.0).toDouble)

    val withWilderness = wildernessAssembler.transform(data).
      drop(wildernessCols:_*).
      withColumn("wilderness", unhotUDF($"wilderness"))

    val soilCols = (0 until 40).map(i => s"Soil_Type_$i").toArray

    val soilAssembler = new VectorAssembler().
      setInputCols(soilCols).
      setOutputCol("soil")

    soilAssembler.transform(withWilderness).
      drop(soilCols:_*).
      withColumn("soil", unhotUDF($"soil"))
  }

// encode the categorical features
val unencTrainData = unencodeOneHot(trainData)
val unencCvData = unencodeOneHot(cvData)
val unencTestData = unencodeOneHot(testData)

val inputCols = unencTrainData.columns.filter(_ != "Cover_Type")
val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featureVector")

val indexer = new VectorIndexer().setMaxCategories(40).setInputCol("featureVector").setOutputCol("indexedVector")

val classifier = new DecisionTreeClassifier().setSeed(12345).setLabelCol("Cover_Type").setFeaturesCol("indexedVector").setPredictionCol("prediction")
val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

// create param grid
val paramGrid = new ParamGridBuilder().addGrid(classifier.impurity, Seq("gini", "entropy")).addGrid(classifier.maxDepth, Seq(1, 20)).addGrid(classifier.maxBins, Seq(40, 300)).addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).build()



val multiclassEval = new MulticlassClassificationEvaluator().setLabelCol("Cover_Type").setPredictionCol("prediction").setMetricName("accuracy")

val validator = new TrainValidationSplit().setSeed(12345).setEstimator(pipeline).setEvaluator(multiclassEval).setEstimatorParamMaps(paramGrid).setTrainRatio(0.9)
val validatorModel = validator.fit(unencTrainData)

val bestModel = validatorModel.bestModel

println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

val testAccuracy = multiclassEval.evaluate(bestModel.transform(unencTestData))
println(testAccuracy)



/// 92%




// random forest classifier
val unencTrainData = unencodeOneHot(trainData)
val unencTestData = unencodeOneHot(testData)
val inputCols = unencTrainData.columns.filter(_ != "Cover_Type")
val assembler = new VectorAssembler().setInputCols(inputCols).setOutputCol("featureVector")

val indexer = new VectorIndexer().setMaxCategories(40).setInputCol("featureVector").setOutputCol("indexedVector")
val classifier = new RandomForestClassifier().setSeed(123456).setLabelCol("Cover_Type").setFeaturesCol("indexedVector").setPredictionCol("prediction").setImpurity("entropy").setMaxDepth(20).setMaxBins(300)

val pipeline = new Pipeline().setStages(Array(assembler, indexer, classifier))

    val paramGrid = new ParamGridBuilder().
      addGrid(classifier.minInfoGain, Seq(0.0, 0.05)).
      addGrid(classifier.numTrees, Seq(1, 10)).
      build()

val multiclassEval = new MulticlassClassificationEvaluator().setLabelCol("Cover_Type").setPredictionCol("prediction").setMetricName("accuracy")

val validator = new TrainValidationSplit().
      setSeed(12345).
      setEstimator(pipeline).
      setEvaluator(multiclassEval).
      setEstimatorParamMaps(paramGrid).
      setTrainRatio(0.9)

val validatorModel = validator.fit(unencTrainData)
val bestModel = validatorModel.bestModel
val forestModel = bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[RandomForestClassificationModel]
println(forestModel.extractParamMap)
println(forestModel.getNumTrees)
forestModel.featureImportances.toArray.zip(inputCols).sorted.reverse.foreach(println)

val testAccuracy = multiclassEval.evaluate(bestModel.transform(unencTestData))
println(testAccuracy)

bestModel.transform(unencTestData.drop("Cover_Type")).select("prediction").show()

//
//
//
