

library(dplyr)
library(h2o)
library(data.table)
packageVersion("h2o")

h2o.init(ip = "localhost", port = 4444, nthreads = 15, max_mem_size = "8g", strict_version_check = FALSE)


# Load datasets
ais <- fread("/Users/mauropelucchi/Desktop/Instacart/aisles.csv", key = "aisle_id")
dept <- fread("/Users/mauropelucchi/Desktop/Instacart/departments.csv", key = "department_id")
prod <- fread("/Users/mauropelucchi/Desktop/Instacart/products.csv", key = c("product_id","aisle_id", "department_id"))
opp <- fread("/Users/mauropelucchi/Desktop/Instacart/order_products__prior.csv")
opt <- fread("/Users/mauropelucchi/Desktop/Instacart/order_products__train.csv")
ord <- fread("/Users/mauropelucchi/Desktop/Instacart/orders.csv")

# Get product department and aisle names
prod <- merge(prod, ais, by="aisle_id", all.x=TRUE, sort=FALSE)
prod <- merge(prod, dept, by="department_id", all.x=TRUE, sort=FALSE)

# For the prior orders get the associated product, aisle, departments, and users
opp <- merge(opp, prod, by="product_id", all.x=TRUE, sort=FALSE)
opp <- merge(opp, ord, by="order_id", all.x=TRUE, sort=FALSE)


ord_max <- opp %>%
  group_by(user_id) %>%
  summarize(order_max = max(order_number), purch_count = n())
head(ord_max)

opp <- left_join(opp, ord_max, by="user_id")
opp <- opp %>% mutate(orders_ago=order_max - order_number + 1)
select(opp, order_number, user_id, order_max, orders_ago, purch_count)

# Create a few simple features

user_prod_list <- opp %>%
  group_by(user_id, product_id) %>%
  summarize(last_order_number = max(order_number), purch_count = n(), avg_hour = mean(order_hour_of_day))

# Compure reordered rate
user_prod_list_reordered <- opp %>%
  filter(reordered == 1) %>%
  group_by(user_id, product_id) %>%
  summarize(reordered_count = n())
user_prod_list <- left_join(user_prod_list, user_prod_list_reordered, by=c("user_id", "product_id"))
user_prod_list <- user_prod_list %>% mutate(reorder_rate=reordered_count/purch_count)
user_prod_list <- user_prod_list %>% mutate(reorder_rate=ifelse(is.na(reorder_rate), 0, reorder_rate))
user_prod_list <- user_prod_list %>% mutate(reordered_count=ifelse(is.na(reordered_count), 0, reordered_count))
head(user_prod_list)

user_summ <- opp %>%
  group_by(user_id) %>%
  summarize(user_total_products_ordered_hist = n(), 
            uniq_prod = n_distinct(product_name),
            uniq_aisle = n_distinct(aisle),
            uniq_dept = n_distinct(department),
            prior_orders = max(order_number),
            avg_hour = mean(order_hour_of_day),
            average_days_between_orders = mean(days_since_prior_order),
            total_order = n_distinct(order_number),
            average_basket = n() / n_distinct(order_number)
  )
head(user_summ)
user_prior_prod_cnt <- opp %>%
  group_by(user_id, product_id) %>%
  summarize(prior_prod_cnt = n(), 
            last_purchased_orders_ago = min(orders_ago),
            first_purchased_orders_ago = max(orders_ago),
            average_days_between_ord_prods = mean(days_since_prior_order)
  )
head(user_prior_prod_cnt)




# Merge datasets to create training frame
opt_user <- left_join(filter(opt, reordered==1), ord, by="order_id")
dt_expanded <- left_join(user_prod_list, opt_user, by=c("user_id", "product_id"))
dt_expanded <- dt_expanded %>% mutate(curr_prod_purchased=ifelse(!is.na(order_id), 1, 0))
#head(dt_expanded)
train <- left_join(dt_expanded, user_summ, by="user_id")
train <- left_join(train, user_prior_prod_cnt, by=c("user_id", "product_id"))
varnames <- setdiff(colnames(train), c("user_id","order_id","curr_prod_purchased"))
head(train)

# Create the test frame
test_orders <- filter(ord, eval_set=="test")
dt_expanded_test <- inner_join(user_prod_list, test_orders, by=c("user_id"))
dt_expanded_test <- dt_expanded_test %>% mutate(curr_prod_purchased=sample(c(0,1), n(), replace=TRUE))
#head(dt_expanded_test)
test <- inner_join(dt_expanded_test, user_summ, by="user_id")
test <- inner_join(test, user_prior_prod_cnt, by=c("user_id", "product_id"))
head(test)

# Check target
test %>% ungroup %>% distinct(curr_prod_purchased)
train %>% ungroup %>% distinct(curr_prod_purchased)

# Sample users for the validation set
set.seed(2222)
unique_user_id <- select(ord_max, user_id)
head(unique_user_id)
val_users <- sample_n(unique_user_id, size=10000, replace=FALSE)
head(val_users)

# Ungroup and convert to factor
train <- train %>% ungroup() %>% mutate(curr_prod_purchased=as.factor(curr_prod_purchased))
test <- test %>% ungroup() %>% mutate(curr_prod_purchased=as.factor(curr_prod_purchased))
test %>% distinct(curr_prod_purchased)
train %>% distinct(curr_prod_purchased)


# Some exploratory analysis

# The top 10 aisles that represent the 45% of sales
library(ggplot2)
tmp <- filter(opp, reordered == 1) %>%
  group_by(aisle,department) %>%
  tally(sort=TRUE) %>%
  mutate(perc = round(100*n/nrow(opp),2)) %>%
  ungroup() %>%
  top_n(10,n)
tmp %>%
  ggplot(aes(x=reorder(aisle, -n), y=n, fill=department)) +
  geom_bar(stat="identity") +
  theme(axis.text.x=element_text(angle=90, hjust=1), axis.title.x = element_blank())# 

# Products vs number of times ordered/reordered
t <- filter(opp) %>% select(product_id, product_name) %>% group_by(product_id, product_name) %>% summarize(ncount=n()) %>% ungroup()
r <- filter(opp, reordered == 1) %>% select(product_id, product_name) %>% group_by(product_id, product_name) %>% summarize(rcount=n()) %>% ungroup()
t <- left_join(t, r, by="product_id") %>% top_n(20,ncount)
t

t %>%
  ggplot() +
  geom_bar(aes(x=reorder(product_name.x , -ncount), y=ncount, fill="Number of times ordered" ), stat="identity") +
  geom_bar(aes(x=reorder(product_name.x , -ncount), y=rcount, fill="Number of times reordered" ), stat="identity") +
  guides(fill=guide_legend(title=element_blank())) +
  theme(axis.text.x=element_text(angle=90, hjust=1), axis.title.x = element_blank(), axis.title.y=element_blank())# 



# Add data to H2O
colnames(train)
select(train, prior_orders)
train_tpl <- anti_join(train, val_users, by="user_id")  %>% select(c("curr_prod_purchased", "user_id", "reordered_count", "product_id", "last_order_number", "purch_count", "avg_hour.x",
                                                                     "order_dow", "add_to_cart_order", "reordered", "user_total_products_ordered_hist", "uniq_prod", "uniq_aisle", "uniq_dept",
                                                                     "prior_orders", "avg_hour.y", "average_days_between_orders", "total_order", "average_basket", "prior_prod_cnt",
                                                                     "last_purchased_orders_ago", "first_purchased_orders_ago", "average_days_between_ord_prods"))
val_tpl <- inner_join(train, val_users, by="user_id")  %>% select(c("curr_prod_purchased", "user_id", "reordered_count", "product_id", "last_order_number", "purch_count", "avg_hour.x",
                                                                    "order_dow", "add_to_cart_order", "reordered", "user_total_products_ordered_hist", "uniq_prod", "uniq_aisle", "uniq_dept",
                                                                    "prior_orders", "avg_hour.y", "average_days_between_orders", "total_order", "average_basket", "prior_prod_cnt",
                                                                    "last_purchased_orders_ago", "first_purchased_orders_ago", "average_days_between_ord_prods"))

train.hex <- as.h2o(train_tpl, destination_frame = "train.hex")
val.hex <- as.h2o(val_tpl, destination_frame = "val.hex")

# Free up some memory
rm(train, opp, opt, ord, prod, dept, ais, user_prod_list, user_summ);gc()
rm(train_tpl, val_tpl);gc();

# Train xgboost model
xgb <- h2o.xgboost(x = c("user_id", "reordered_count", "product_id", "last_order_number", "purch_count", "avg_hour.x",
                         "order_dow", "add_to_cart_order", "reordered", "user_total_products_ordered_hist", "uniq_prod", "uniq_aisle", "uniq_dept",
                         "prior_orders", "avg_hour.y", "average_days_between_orders", "total_order", "average_basket", "prior_prod_cnt",
                         "last_purchased_orders_ago", "first_purchased_orders_ago", "average_days_between_ord_prods")
                   ,y = "curr_prod_purchased"
                   ,training_frame = train.hex
                   ,validation_frame = val.hex
                   ,model_id = "xgb_model_1"
                   ,stopping_rounds = 3
                   ,stopping_metric = "logloss"
                   ,distribution = "bernoulli"
                   ,score_tree_interval = 1
                   ,learn_rate=0.1
                   ,ntrees=20
                   ,subsample = 0.75
                   ,colsample_bytree = 0.75
                   ,tree_method = "hist"
                   ,grow_policy = "lossguide"
                   ,booster = "gbtree"
                   ,gamma = 0.0
)


# Make predictions
test.hex <- as.h2o(test, destination_frame = "test.hex")
predictions <- as.data.table(h2o.predict(xgb, test.hex))

predictions <- data.table(order_id=test$order_id, product_id=test$product_id, testPreds=predictions$predict, p0=predictions$p0, p1=predictions$p1)
filter(predictions, testPreds==1)
testPreds <- predictions[,.(products=paste0(product_id[p0>0.21], collapse=" ")), by=order_id]
set(testPreds, which(testPreds[["products"]]==""), "products", "None")
# Create submission file
fwrite(testPreds, "/Users/mauropelucchi/Desktop/Instacart/submission.csv")
