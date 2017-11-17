# Copyright 2017 Mauro Pelucchi
#
# See LICENSE file for further information.

# setup spark and R env (first time only)
install.packages('devtools')
devtools::install_github('apache/spark@v2.2.0', subdir='R/pkg')
install.packages('sparklyr')
install.packages("rgl")

# load library
library(sparklyr)
library(rgl)

# set spark and java home
Sys.setenv(SPARK_HOME = "/Users/mauropelucchi/Desktop/My_Home/Tools/spark-2.2.0-bin-hadoop2.7")
# set java home to Java 1.8
Sys.setenv(JAVA_HOME = "/Library/Java/JavaVirtualMachines/jdk1.8.0_111.jdk/Contents/Home")
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))


# Set this appropriately for your cluster
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "4g"))

clusters_data <- read.df("/Users/mauropelucchi/Desktop/Machine_Learning/Dataset/KDD1999/kddcup.data", "csv",
                         inferSchema = "true", header = "false")

colnames(clusters_data) <- c(
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

# read only numeric columns
numeric_only <- cache(drop(clusters_data,c("protocol_type", "service", "flag", "label")))

# create kmeans model with k = 100, train the models with numeric_only features
kmeans_model <- spark.kmeans(numeric_only, ~ ., k = 100, maxIter = 40, initMode = "k-means||")
# apply clustering and make predictions
clustering <- predict(kmeans_model, numeric_only)
# sample 20% of results
clustering_sample <- collect(sample(clustering, FALSE, 0.05))
str(clustering_sample)
# extract predicition columns
clusters <- clustering_sample["prediction"]
data <- data.matrix(within(clustering_sample, rm("prediction")))
table(data)

# make a random 3D projection and normalize
random_projection <- matrix(data = rnorm(3*ncol(data)), ncol = 3)
random_projection_norm <- random_projection / sqrt(rowSums(random_projection*random_projection))

# project and make a new data frame
projected_data <- data.frame(data %*% random_projection_norm)

num_clusters <- max(clusters)
# 97
palette <- rainbow(num_clusters)
colors = sapply(clusters, function(c) palette[c])
# plot 3d data
plot3d(projected_data, col = colors, size = 10)

unpersist(numeric_only)