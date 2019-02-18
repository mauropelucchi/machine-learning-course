# to install h2o
wget http://h2o-release.s3.amazonaws.com/h2o/rel-xu/3/h2o-3.22.1.3.zip
unzip h2o-3.22.1.3.zip
cd h2o-3.22.1.3/R
pwd

install.packages("/home/master/h2o-3.22.1.3/h2o-3.22.1.3.tar.gz",repos = NULL, type = "source")
library(h2o)
demo(h2o.glm)

# or

install.packages("h20")
library(h2o)
demo(h2o.glm)


# Uber and H2O clustering

library(dplyr)
library(h2o)
library(lubridate)
library(DT)
library(ggmap)


# Load the .csv files
#may14 <- read.csv("/home/master/dataset/uber/uber-raw-data-may14.csv")
#apr14 <- read.csv("/home/master/dataset/uber/uber-raw-data-apr14.csv")
#jun14 <- read.csv("/home/master/dataset/uber/uber-raw-data-jun14.csv")
#jul14 <- read.csv("/home/master/dataset/uber/uber-raw-data-jul14.csv")
#aug14 <- read.csv("/home/master/dataset/uber/uber-raw-data-aug14.csv")
sep14 <- read.csv("/home/master/dataset/uber/uber-raw-data-sep14.csv")


# Union of all dataset
#data14 <- bind_rows(apr14, may14, jun14, jul14, aug14, sep14)
data14 <- bind_rows(sep14)
summary(data14)



# Data preparation

# format date/time column
data14$Date.Time <- mdy_hms(data14$Date.Time)
data14$Year <- factor(year(data14$Date.Time))
data14$Month <- factor(month(data14$Date.Time))
data14$Day <- factor(day(data14$Date.Time))
data14$Weekday <- factor(wday(data14$Date.Time))
data14$Hour <- factor(hour(data14$Date.Time))
data14$Minute <- factor(minute(data14$Date.Time))
data14$Second <- factor(second(data14$Date.Time))
data14$Date.Time <- as.Date(data14$Date.Time)

head(data14, n=10)


set.seed(20)

# load h2o
library(h2o)
# initialize h2o instance
h2o.init()
# upload the dataframe
hf <- as.h2o(data14, key="data14.hex")

# clustering
km2 <- h2o.kmeans(hf, x = colnames(data14)[2:3], k = 5)
km2
p <- h2o.predict(km2, hf)

# set the cluster number in the dataset as column 'Borough'
clusters <- as.data.frame(p)
data14$Borough <- clusters$predict


# show the map
library(ggmap)
register_google(key = "****************************************")
NYCMap <- get_map("New York", zoom = 10)
ggmap(NYCMap) + geom_point(aes(x = Lon[], y = Lat[], colour = as.factor(Borough)),data = data14) +
  ggtitle("NYC Boroughs using KMean")



data14$Month <- as.double(data14$Month)
month_borough_14 <- count_(data14, vars = c('Month', 'Borough'), sort = TRUE) %>% 
  arrange(Month, Borough)
datatable(month_borough_14)

monthly_growth <- month_borough_14 %>%
  mutate(Date = paste("04", Month)) %>%
  ggplot(aes(Month, n, colour = Borough)) + geom_line() +
  ggtitle("Uber Monthly Growth - 2014")
monthly_growth
