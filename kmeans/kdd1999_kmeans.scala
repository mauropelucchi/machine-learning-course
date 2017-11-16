/*
 * Copyright 2017 Mauro Pelucchi
 *
 * See LICENSE file for further information.
 */


//
// bin/spark-shell --driver-memory=4g
// 
// Detect Network Anomalies with K-means
// Data from KDD 1999 CUp
//
// Data from http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
// 


import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.feature.{OneHotEncoder, VectorAssembler, StringIndexer, StandardScaler}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import scala.util.Random

//
// read data from KDD 1999 CUP DATASET
//
val data = spark.read.
      option("inferSchema", true).
      option("header", false).
      csv("/Users/mauropelucchi/Desktop/Machine_Learning/Dataset/KDD1999/kddcup.data").
      toDF(
        "duration", "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds",
        "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
        "dst_host_count", "dst_host_srv_count",
        "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
        "label")

data.cache
data.count

// about 4,898,431.0 records

data.printSchema


// explore label of connections (for supervised learning classifier)
data.select("label").groupBy("label").count().orderBy($"count".desc).show(false)


//
//+----------------+-------+
//|label           |count  |
//+----------------+-------+
//|smurf.          |2807886|
//|neptune.        |1072017|
//|normal.         |972781 |
//|satan.          |15892  |
//|ipsweep.        |12481  |
//|portsweep.      |10413  |
//|nmap.           |2316   |
//|back.           |2203   |
//|warezclient.    |1020   |
//|teardrop.       |979    |
//|pod.            |264    |
//|guess_passwd.   |53     |
//|buffer_overflow.|30     |
//|land.           |21     |
//|warezmaster.    |20     |
//|imap.           |12     |
//|rootkit.        |10     |
//|loadmodule.     |9      |
//|ftp_write.      |8      |
//|multihop.       |7      |
//+----------------+-------+


// cache numerical features and assembly vector with all features
val numericOnly = data.drop("protocol_type", "service", "flag").cache()
val assembler = new VectorAssembler().setInputCols(numericOnly.columns.filter(_ != "label")).setOutputCol("featureVector")

// train first KMEAS models
val kmeans = new KMeans().setSeed(12345).setPredictionCol("cluster").setFeaturesCol("featureVector")

// define pipeplie
// assembler --> kmeas
val pipeline = new Pipeline().setStages(Array(assembler, kmeans))
val pipelineModel = pipeline.fit(numericOnly)
val kmeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]

kmeansModel.clusterCenters.foreach(println)



// describe data from label and cluster columns
val withCluster = pipelineModel.transform(numericOnly)

withCluster.select("cluster", "label").groupBy("cluster", "label").count().orderBy($"cluster", $"count".desc).show(25)
numericOnly.unpersist()

import spark.implicits._


//
//+-------+----------------+-------+                                              
//|cluster|           label|  count|
//+-------+----------------+-------+
//|      0|          smurf.|2807886|
//|      0|        neptune.|1072017|
//|      0|         normal.| 972781|
//|      0|          satan.|  15892|
//|      0|        ipsweep.|  12481|
//|      0|      portsweep.|  10412|
//|      0|           nmap.|   2316|
//|      0|           back.|   2203|
//|      0|    warezclient.|   1020|
//|      0|       teardrop.|    979|
//|      0|            pod.|    264|
//|      0|   guess_passwd.|     53|
//|      0|buffer_overflow.|     30|
//|      0|           land.|     21|
//|      0|    warezmaster.|     20|
//|      0|           imap.|     12|
//|      0|        rootkit.|     10|
//|      0|     loadmodule.|      9|
//|      0|      ftp_write.|      8|
//|      0|       multihop.|      7|
//|      0|            phf.|      4|
//|      0|           perl.|      3|
//|      0|            spy.|      2|
//|      1|      portsweep.|      1|
//+-------+----------------+-------+




// one hot pipeline encoder
def oneHotPipeline(inputCol: String): (Pipeline, String) = {
    val indexer = new StringIndexer().
      setInputCol(inputCol).
      setOutputCol(inputCol + "_indexed")
    val encoder = new OneHotEncoder().
      setInputCol(inputCol + "_indexed").
      setOutputCol(inputCol + "_vec")
    val pipeline = new Pipeline().setStages(Array(indexer, encoder))
    (pipeline, inputCol + "_vec")
  }




// choose k
def fitPipeline(data: DataFrame, k: Int): PipelineModel = {
    val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
    val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
    val (flagEncoder, flagVecCol) = oneHotPipeline("flag")

    // Original columns, without label / string columns, but with new vector encoded cols
    val assembleCols = Set(data.columns: _*) --
      Seq("label", "protocol_type", "service", "flag") ++
      Seq(protoTypeVecCol, serviceVecCol, flagVecCol)
    val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

    val scaler = new StandardScaler()
      .setInputCol("featureVector")
      .setOutputCol("scaledFeatureVector")
      .setWithStd(true)
      .setWithMean(false)

    val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

    val pipeline = new Pipeline().setStages(
      Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
    pipeline.fit(data)
  }

 def clusteringScore(data: DataFrame, k: Int): Double = {
    val pipelineModel = fitPipeline(data, k)

    // Predict cluster for each datum
    val clusterLabel = pipelineModel.transform(data).
      select("cluster", "label").as[(Int, String)]
    val weightedClusterEntropy = clusterLabel.
      // Extract collections of labels, per cluster
      groupByKey { case (cluster, _) => cluster }.
      mapGroups { case (_, clusterLabels) =>
        val labels = clusterLabels.map { case (_, label) => label }.toSeq
        // Count labels in collections
        val labelCounts = labels.groupBy(identity).values.map(_.size)
        labels.size * entropy(labelCounts)
      }.collect()

    // Average entropy weighted by cluster size
    weightedClusterEntropy.sum / data.count()
  }

(60 to 270 by 30).map(k => (k, clusteringScore(data, k))).foreach(println)




// final model
// encode categoricals labels
val (protoTypeEncoder, protoTypeVecCol) = oneHotPipeline("protocol_type")
val (serviceEncoder, serviceVecCol) = oneHotPipeline("service")
val (flagEncoder, flagVecCol) = oneHotPipeline("flag")


// Original columns, without label / string columns, but with new vector encoded cols
val assembleCols = Set(data.columns: _*) -- Seq("label", "protocol_type", "service", "flag") ++ Seq(protoTypeVecCol, serviceVecCol, flagVecCol)

val assembler = new VectorAssembler().
      setInputCols(assembleCols.toArray).
      setOutputCol("featureVector")

val scaler = new StandardScaler().setInputCol("featureVector").setOutputCol("scaledFeatureVector").setWithStd(true).setWithMean(false)

val k = 180
val kmeans = new KMeans().
      setSeed(Random.nextLong()).
      setK(k).
      setPredictionCol("cluster").
      setFeaturesCol("scaledFeatureVector").
      setMaxIter(40).
      setTol(1.0e-5)

val pipeline = new Pipeline().setStages(Array(protoTypeEncoder, serviceEncoder, flagEncoder, assembler, scaler, kmeans))
val pipelineModel = pipeline.fit(data)


val kMeansModel = pipelineModel.stages.last.asInstanceOf[KMeansModel]
val centroids = kMeansModel.clusterCenters

// cluster data
val clustered = pipelineModel.transform(data)
val threshold = clustered.select("cluster", "scaledFeatureVector").as[(Int, Vector)].map { case (cluster, vec) => Vectors.sqdist(centroids(cluster), vec) }.orderBy($"value".desc).take(100).last

val originalCols = data.columns
val anomalies = clustered.filter { row =>
  val cluster = row.getAs[Int]("cluster")
  val vec = row.getAs[Vector]("scaledFeatureVector")
  Vectors.sqdist(centroids(cluster), vec) >= threshold
}.select(originalCols.head, originalCols.tail:_*)

println(anomalies.first())
