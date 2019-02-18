#
# Setup Spark2
#
Sys.setenv(SPARK_HOME = "/home/master/Desktop/spark-2.4.0-bin-hadoop2.7")
Sys.setenv(SPARK_LOCAL_IP='127.0.0.1')
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))
sparkR.session(master = "local[*]", sparkConfig = list(spark.driver.memory = "2g"))

sc = sparkR.init()

#
# word count
#
lines = SparkR:::textFile(sc,"/home/master/dataset/dc.txt")


SparkR:::countRDD(lines)

words = SparkR:::flatMap(lines, function(line) {
  strsplit(line," ")[[1]]
})
wordCount = SparkR:::lapply(words, function(word) {
         list(word,1L)
       })
counts = SparkR:::reduceByKey(wordCount, "+", 2L)
output = SparkR:::collect(as.DataFrame(counts))


#
# Uber Data on SparkR
#

# read the csv
uber_aug14 = read.df("/home/master/dataset/uber/uber-raw-data-aug14.csv", source="csv", header=TRUE)
uber_sep14 = read.df("/home/master/dataset/uber/uber-raw-data-sep14.csv", source="csv", header=TRUE)
printSchema(uber_sep14)
# count
SparkR:::count(uber_sep14)

# union of the dataset
uber_2014 = rbind(uber_aug14, uber_sep14)
SparkR:::count(uber_2014)
# rename the column
colnames(uber_2014) = c("Date_Time","Lat","Lon","Base")
printSchema(uber_2014)

# explore the dataset
library(tidyverse)
head(uber_2014)

# format date and time
uber_2014 = uber_2014 %>% withColumn("Date", to_date(.$Date_Time,  "M/d/yyyy"))
head(uber_2014)
uber_2014 = uber_2014 %>% withColumn("Time", to_timestamp(.$Date_Time,  "M/d/yyyy H:mm:ss"))
head(uber_2014, 1000)


uber_2014 = uber_2014 %>% withColumn("Day", dayofmonth(.$Date))
uber_2014 = uber_2014 %>% withColumn("Month", month(.$Date))
uber_2014 = uber_2014 %>% withColumn("Year", year(.$Date))
uber_2014 = uber_2014 %>% withColumn("WeekDay", dayofweek(.$Date))
uber_2014 = uber_2014 %>% withColumn("Hour", hour(.$Time))
uber_2014 = uber_2014 %>% withColumn("Minute", minute(.$Time))
head(uber_2014)

### Number of Pickups by Day
by_day <- SparkR:::count(groupBy(uber_2014, uber_2014$Day))
head(by_day)
SparkR:::count(by_day)

library(ggplot2)
library(DT)
datatable(SparkR:::collect(by_day))


ggplot(SparkR:::collect(by_day), aes(Day, count)) + 
  geom_bar( stat = "identity", fill = "darkred") +
  ggtitle("Trips Every Day") +
  theme(legend.position = "none")

by_month_day <- SparkR:::count(groupBy(uber_2014, uber_2014$Month, uber_2014$Day))
head(by_month_day)

ggplot(SparkR:::collect(SparkR:::orderBy(by_month_day, by_month_day$Month, by_month_day$Day)), aes(Day, count, fill = Month)) + 
  geom_bar( stat = "identity") +
  ggtitle("Trips by Day and Month")



### Number of Trips by Month
by_month <- SparkR:::count(groupBy(uber_2014, uber_2014$Month))
datatable(SparkR:::collect(by_month))

ggplot(SparkR:::collect(by_month), aes(Month, count, fill = Month)) + 
  geom_bar( stat = "identity") +
  ggtitle("Trips by Month") +
  theme(legend.position = "none")

by_month_weekday <- SparkR:::count(groupBy(uber_2014, uber_2014$Month, uber_2014$WeekDay))
by_month_weekday <- SparkR:::orderBy(by_month_weekday, by_month_weekday$Month, by_month_weekday$WeekDay)
head(by_month_weekday, 20)
ggplot(SparkR:::collect(by_month_weekday), aes(WeekDay, Month, fill = count)) + 
  geom_tile(color = "white") +
  ggtitle("Trips by Day and Month")




### Heat Map of Hour, Day
by_hour_day <- SparkR:::count(groupBy(uber_2014, uber_2014$WeekDay, uber_2014$Hour))
datatable(SparkR:::collect(by_hour_day))

ggplot(SparkR:::collect(by_hour_day), aes(WeekDay, Hour, fill = count)) +
  geom_tile(color = "white") +
  ggtitle("Heat Map by Hour and Day")




# load other months
uber_aug14 = read.df("/home/master/dataset/uber/uber-raw-data-aug14.csv", source="csv", header=TRUE)
uber_sep14 = read.df("/home/master/dataset/uber/uber-raw-data-sep14.csv", source="csv", header=TRUE)
uber_jul14 = read.df("/home/master/dataset/uber/uber-raw-data-jul14.csv", source="csv", header=TRUE)
uber_jun14 = read.df("/home/master/dataset/uber/uber-raw-data-jun14.csv", source="csv", header=TRUE)
# union of the dataset
uber_2014 = rbind(uber_aug14, uber_sep14, uber_jul14, uber_jun14)
SparkR:::count(uber_2014)
# rename the column
colnames(uber_2014) = c("Date_Time","Lat","Lon","Base")
printSchema(uber_2014)

# format date and time
uber_2014 = uber_2014 %>% withColumn("Date", to_date(.$Date_Time,  "M/d/yyyy"))
uber_2014 = uber_2014 %>% withColumn("Time", to_timestamp(.$Date_Time,  "M/d/yyyy H:mm:ss"))
uber_2014 = uber_2014 %>% withColumn("Day", dayofmonth(.$Date))
uber_2014 = uber_2014 %>% withColumn("Month", month(.$Date))
uber_2014 = uber_2014 %>% withColumn("Year", year(.$Date))
uber_2014 = uber_2014 %>% withColumn("WeekDay", dayofweek(.$Date))
uber_2014 = uber_2014 %>% withColumn("Hour", hour(.$Time))
uber_2014 = uber_2014 %>% withColumn("Minute", minute(.$Time))
head(uber_2014)


### Heat Map of Hour, Day
by_hour_day <- SparkR:::count(groupBy(uber_2014, uber_2014$WeekDay, uber_2014$Hour))
datatable(SparkR:::collect(by_hour_day))

ggplot(SparkR:::collect(by_hour_day), aes(WeekDay, Hour, fill = count)) +
  geom_tile(color = "white") +
  ggtitle("Heat Map by Hour and Day")