# Kaggle
# Toxic Comment Classification Challenge
# Identify and classify toxic online comments
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# toxic, severe_toxic, obscene ,threat ,insult ,identity_hate
#
# H2O and Text mining for Toxic Comment Classification Challenge
#
# Full script at https://github.com/mauropelucchi/machine-learning-course/blob/master/text-mining/h2o_toxic_comments.r
#

library(data.table)
library(dplyr)
library(h2o)
library(stringr)
library(ngram)
library(tokenizers)
packageVersion("h2o")

h2o.init(ip = "localhost", port = 4444, nthreads = -1, max_mem_size = "4g", strict_version_check = FALSE)

train <- fread("/Users/mauropelucchi/Desktop/Toxic_comments/train.csv", key=c("id"))
test <- fread("/Users/mauropelucchi/Desktop/Toxic_comments/test.csv", key=c("id"))
stopwords.en <- fread("/Users/mauropelucchi/Desktop/Toxic_comments/stopwords-en.txt")

# drop duplicate --> no duplicate to remove
nrow(train)
train <- train %>% mutate(filter="train")#  %>% distinct("comment_text", "filter", "toxic", "severe_toxic", "obscene","threat" ,"insult" ,"identity_hate")
test <- test %>% mutate(filter="test")

all_comments <- train %>% bind_rows(test)
nrow(all_comments)

# Remove all special chars
all_comments <- all_comments %>% mutate(clean_text=str_replace_all(comment_text, "[^[:alnum:]]", " "))

# Go to H2O
comments.hex <- as.h2o(select(all_comments, c("id", "clean_text", "filter", "toxic", "severe_toxic", "obscene","threat" ,"insult" ,"identity_hate")), destination_frame = "all_comments.hex", col.types=c("String"))

stop_words_chr <- c(stopwords.en$C1) %>% c('wikipedia','username')

# Tokenize, lowercase, remove text with len less than 3 chars
tokenize <- function(text) {
  tokenized <- h2o.tokenize(text, "\\\\W+")
  tokenized.lower <- h2o.tolower(tokenized)
  # remove (less than 2 chars)
  tokenized.lengths <- h2o.nchar(tokenized.lower)
  tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths > 2,]
  # remove words that contain numbers
  tokenized.words <- tokenized.filtered[h2o.grep("[0-9]", tokenized.filtered, invert = TRUE, output.logical = TRUE),]
  tokenized.words <- tokenized.words[is.na(tokenized.words) || (! tokenized.words %in% stop_words_chr),]
}


# Get a list of terms and create stop_words and rare_words lists
words.hex <- tokenize(comments.hex$clean_text)
h2o.head(words.hex)
words <- as.data.frame(words.hex)
words <- words %>% group_by(C1) %>% mutate(wc=n())

stop_words <- words %>% tally(wc) %>% top_n(200) %>% select(C1) %>% ungroup()
rare_words <- words %>% filter(wc < 10) %>% select(C1) %>% ungroup()
rm(words)
h2o.rm(words.hex)
#
stop_words_chr <- c(stop_words$C1) %>% c(stopwords.en$C1) %>% c('wikipedia','username')
rare_words_chr <- c(rare_words$C1)


# Tokenize, lowercase, remove text with len less than 3 chars
clean_token <- function(text) {
  tokenized <- h2o.tokenize(text, "\\\\W+")
  tokenized.lower <- h2o.tolower(tokenized)
  # remove (less than 2 chars)
  tokenized.lengths <- h2o.nchar(tokenized.lower)
  tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths > 2,]
  # remove words that contain numbers
  tokenized.words <- tokenized.filtered[h2o.grep("[0-9]", tokenized.filtered, invert = TRUE, output.logical = TRUE),]
  tokenized.words <- tokenized.words[is.na(tokenized.words) || (! tokenized.words %in% stop_words_chr),]
  tokenized.words <- tokenized.words[is.na(tokenized.words) || (! tokenized.words %in% rare_words_chr),]
}

clean_words.hex <- clean_token(comments.hex$clean_text)
h2o.head(clean_words.hex)
nrow(clean_words.hex)


####################################################################################
# Build word2vec model
vectors <- 10
w2v.model <- h2o.word2vec(clean_words.hex
                          , model_id = "w2v_model"
                          , vec_size = vectors
                          , min_word_freq = 10
                          , window_size = 5
                          , init_learning_rate = 0.025
                          , sent_sample_rate = 0
                          , epochs = 15)


##### Check - find synonyms for the word 'water',mexicans','hardcore','metallica','hot','idiot'
print(h2o.findSynonyms(w2v.model, "water", count = 5))
print(h2o.findSynonyms(w2v.model, "hardcore", count = 5))
print(h2o.findSynonyms(w2v.model, "metallica", count = 5))
print(h2o.findSynonyms(w2v.model, "hot", count = 5))
print(h2o.findSynonyms(w2v.model, "idiot", count = 5))
print(h2o.findSynonyms(w2v.model, "mexicans", count = 5))



#### Calculate a vector for each comment
comment_all.vecs <- h2o.transform(w2v.model, clean_words.hex, aggregate_method = "AVERAGE")


##### Prepare training&validation data
train.set <- comments.hex[, "filter"] == "train"
test.set <- comments.hex[, "filter"] == "test"
##### Factorize target
names(comments.hex)
data <- h2o.cbind(h2o.asfactor(comments.hex[, c("filter","toxic","severe_toxic", "obscene","threat","insult","identity_hate")]), comment_all.vecs)

nrow(comment_all.vecs)
train.hex <- data[train.set,]
test.hex <- data[test.set,]
nrow(train.hex)
nrow(test.hex)

# Models Selection
# Build models for Toxic

########################################################################
########################################################################
########################################################################
#
# GBM models
#

########################################################################
# GBM hyperparamters
gbm.params <- list(learn_rate = c(0.01, 0.1),
                    max_depth = c(3, 9, 15),
                    sample_rate = c(0.8, 1.0),
                    col_sample_rate = c(0.2, 0.5, 1.0)
                   )

# Grid
gbm.grid <- h2o.grid("gbm", x = names(comment_all.vecs), y = "toxic",
                    grid_id = "gbm_grid",
                    training_frame = train.hex,
                    seed = 1,
                    ntrees = 100,
                    nfolds = 10,
                    stopping_metric="AUC",
                    distribution = "bernoulli",
                    hyper_params = gbm.params)

# Get the grid results, sorted by validation AUC
gbm.gridperf <- h2o.getGrid(grid_id = "gbm_grid",
                    sort_by = "auc",
                    decreasing = TRUE)
print(gbm.gridperf)

# Grab the top GBM model, chosen by validation AUC
toxic.model <- h2o.getModel(gbm.gridperf@model_ids[[1]])

# Look at the hyperparamters for the best model
print(toxic.model@model[["model_summary"]])
########################################################################

toxic.model <- h2o.gbm(x = names(comment_all.vecs), y = "toxic",model_id = "toxic_model",
                      training_frame = train.hex, nfolds = 10, ntrees = 100,
                      fold_assignment = "Stratified", sample_rate = 1.0, col_sample_rate = 0.2,
                      stopping_rounds = 3, learn_rate = 0.01, max_depth = 15,
                      stopping_metric="AUC", distribution = "bernoulli")

severe_toxic.model <- h2o.gbm(x = names(comment_all.vecs), y = "severe_toxic",model_id = "severe_toxic_model",
                        training_frame = train.hex, nfolds = 10, ntrees = 100,
                        fold_assignment = "Stratified", sample_rate = 1.0, col_sample_rate = 0.2,
                        stopping_rounds = 3, learn_rate = 0.01, max_depth = 15,
                        stopping_metric="AUC", distribution = "bernoulli")

obscene.model <- h2o.gbm(x = names(comment_all.vecs), y = "obscene",model_id = "obscene_model",
                        training_frame = train.hex, nfolds = 10, ntrees = 100,
                        fold_assignment = "Stratified", sample_rate = 1.0, col_sample_rate = 0.2,
                        stopping_rounds = 3, learn_rate = 0.01, max_depth = 15,
                        stopping_metric="AUC", distribution = "bernoulli")

threat.model <- h2o.gbm(x = names(comment_all.vecs), y = "threat",model_id = "threat_model",
                        training_frame = train.hex, nfolds = 10, ntrees = 100,
                        fold_assignment = "Stratified", sample_rate = 1.0, col_sample_rate = 0.2,
                        stopping_rounds = 3, learn_rate = 0.01, max_depth = 15,
                        stopping_metric="AUC", distribution = "bernoulli")

insult.model <- h2o.gbm(x = names(comment_all.vecs), y = "insult",model_id = "insult_model",
                        training_frame = train.hex, nfolds = 10, ntrees = 100,
                        fold_assignment = "Stratified", sample_rate = 1.0, col_sample_rate = 0.2,
                        stopping_rounds = 3, learn_rate = 0.01, max_depth = 15,
                        stopping_metric="AUC", distribution = "bernoulli")

identity_hate.model <- h2o.gbm(x = names(comment_all.vecs), y = "identity_hate",model_id = "identity_hate_model",
                        training_frame = train.hex, nfolds = 10, ntrees = 100,
                        fold_assignment = "Stratified", sample_rate = 1.0, col_sample_rate = 0.2,
                        stopping_rounds = 3, learn_rate = 0.01, max_depth = 15,
                        stopping_metric="AUC", distribution = "bernoulli")
                               
# Prediction with GBM models

toxic.hex <- h2o.predict(toxic.model , test.hex)
severe_toxic.hex <- h2o.predict(severe_toxic.model, test.hex)
obscene.hex <- h2o.predict(obscene.model, test.hex)
threat.hex <- h2o.predict(threat.model, test.hex)
insult.hex <- h2o.predict(insult.model, test.hex)
identity_hate.hex <- h2o.predict(identity_hate.model, test.hex)


subm <- data.frame(id = test$id)
subm[, "toxic"] <- as.data.table(toxic.hex[,"p1"])
subm[, "severe_toxic"] <- as.data.table(severe_toxic.hex[,"p1"])
subm[, "obscene"] <- as.data.table(obscene.hex[,"p1"])
subm[, "threat"] <- as.data.table(threat.hex[,"p1"])
subm[, "insult"] <- as.data.table(insult.hex[,"p1"])
subm[, "identity_hate"] <- as.data.table(identity_hate.hex[,"p1"])

head(subm)
nrow(subm) #153164
write.csv(subm, '/Users/mauropelucchi/Desktop/Toxic_comments/submission.csv', row.names = FALSE)


#
# Score on Kaggle Leaderboard  0.9235
#
########################################################################
########################################################################
########################################################################


########################################################################
########################################################################
########################################################################
#
# GLM models
#

########################################################################
# GLM hyperparamters
glm.params <- list(lambda = c(0.01, 0.1, 0.5),
                   alpha = c(0, 0.1, 0.5, 1.0)
)

# Grid
glm.grid <- h2o.grid("glm", x = names(comment_all.vecs), y = "toxic",
                     grid_id = "glm_grid",
                     standardize = TRUE,
                     training_frame = train.hex,
                     seed = 2345,
                     nfolds = 10,
                     family = "binomial",
                     hyper_params = glm.params)

# Get the grid results, sorted by validation AUC
glm.gridperf <- h2o.getGrid(grid_id = "glm_grid",
                            sort_by = "auc",
                            decreasing = TRUE)
print(glm.gridperf)

# Grab the top GLM model, chosen by validation AUC
toxic.model <- h2o.getModel(glm.gridperf@model_ids[[1]])

# Look at the hyperparamters for the best model
print(toxic.model@model[["model_summary"]])
########################################################################


toxic.model <- h2o.glm(x = names(comment_all.vecs), y = "toxic",model_id = "toxic_model",
                       training_frame = train.hex, nfolds = 10, standardize = TRUE,
                       lambda=0.1, alpha=0, family = "binomial")

severe_toxic.model <- h2o.glm(x = names(comment_all.vecs), y = "severe_toxic",model_id = "severe_toxic_model",
                       training_frame = train.hex, nfolds = 10, standardize = TRUE,
                       lambda=0.1, alpha=0, family = "binomial")

obscene.model <- h2o.glm(x = names(comment_all.vecs), y = "obscene",model_id = "obscene_model",
                       training_frame = train.hex, nfolds = 10, standardize = TRUE,
                       lambda=0.1, alpha=0, family = "binomial")

threat.model <- h2o.glm(x = names(comment_all.vecs), y = "threat",model_id = "threat_model",
                       training_frame = train.hex, nfolds = 10, standardize = TRUE,
                       lambda=0.1, alpha=0, family = "binomial")

insult.model <- h2o.glm(x = names(comment_all.vecs), y = "insult",model_id = "insult_model",
                        training_frame = train.hex, nfolds = 10, standardize = TRUE,
                        lambda=0.1, alpha=0, family = "binomial")

identity_hate.model <- h2o.glm(x = names(comment_all.vecs), y = "identity_hate",model_id = "identity_hate_model",
                        training_frame = train.hex, nfolds = 10, standardize = TRUE,
                        lambda=0.1, alpha=0, family = "binomial")

# Prediction with H2O AutoML (H2O Auto ML with Stacked Ensemble)
#

identity_hate.hex <- h2o.predict(identity_hate.model, test.hex)
insult.hex <- h2o.predict(insult.model, test.hex)
threat.hex <- h2o.predict(threat.model, test.hex)
severe_toxic.hex <- h2o.predict(severe_toxic.model, test.hex)
toxic.hex <- h2o.predict(toxic.model , test.hex)
obscene.hex <- h2o.predict(obscene.model, test.hex)


subm <- data.frame(id = test$id)
subm[, "toxic"] <- as.data.table(toxic.hex[,"p1"])
subm[, "severe_toxic"] <- as.data.table(severe_toxic.hex[,"p1"])
subm[, "obscene"] <- as.data.table(obscene.hex[,"p1"])
subm[, "threat"] <- as.data.table(threat.hex[,"p1"])
subm[, "insult"] <- as.data.table(insult.hex[,"p1"])
subm[, "identity_hate"] <- as.data.table(identity_hate.hex[,"p1"])

head(subm)
nrow(subm) #153164
write.csv(subm, '/Users/mauropelucchi/Desktop/Toxic_comments/submission.csv', row.names = FALSE)

#
# Score on Kaggle Leaderboard 0.9112
#
########################################################################
########################################################################
########################################################################


########################################################################
########################################################################
########################################################################
#### H2O Auto ML

identity_hate.model = h2o.deeplearning(x= names(comment_all.vecs), 
                         y= "identity_hate", 
                         training_frame=train.hex, 
                         nfolds = 10, 
                         fold_assignment = "Stratified", 
                         distribution = "bernoulli",
                         activation = "RectifierWithDropout",
                         hidden = c(10,10,10,10),
                         input_dropout_ratio = 0.2,
                         l1 = 1e-5,
                         epochs = 50)

identity_hate.model = h2o.automl(x= names(comment_all.vecs), 
                         y= "identity_hate", 
                         training_frame=train.hex, 
                         nfolds = 10, 
                         max_models = 10,
                         max_runtime_secs = 3600,
                         stopping_metric  = "AUC",
                         stopping_rounds = 3)

toxic.model = h2o.automl(x= names(comment_all.vecs), 
                         y= "toxic", 
                         training_frame=train.hex, 
                         nfolds = 10, 
                         max_models = 10,
                         max_runtime_secs = 3600,
                         stopping_metric  = "AUC",
                         stopping_rounds = 3)

severe_toxic.model = h2o.automl(x= names(comment_all.vecs), 
                         y= "severe_toxic", 
                         training_frame=train.hex, 
                         nfolds = 10, 
                         max_models = 10,
                         max_runtime_secs = 3600,
                         stopping_metric  = "AUC",
                         stopping_rounds = 3)

obscene.model = h2o.automl(x= names(comment_all.vecs), 
                         y= "obscene", 
                         training_frame=train.hex, 
                         nfolds = 10, 
                         max_models = 10,
                         max_runtime_secs = 3600,
                         stopping_metric  = "AUC",
                         stopping_rounds = 3)

insult.model = h2o.automl(x= names(comment_all.vecs), 
                         y= "insult", 
                         training_frame=train.hex, 
                         nfolds = 10, 
                         max_models = 10,
                         max_runtime_secs = 3600,
                         stopping_metric  = "AUC",
                         stopping_rounds = 3)

threat.model = h2o.automl(x= names(comment_all.vecs), 
                         y= "threat", 
                         training_frame=train.hex, 
                         nfolds = 10, 
                         max_models = 10,
                         max_runtime_secs = 3600,
                         stopping_metric  = "AUC",
                         stopping_rounds = 3)

# Prediction with H2O AutoML (H2O Auto ML with Stacked Ensemble)
#

identity_hate.hex <- h2o.predict(identity_hate.model, test.hex)
insult.hex <- h2o.predict(insult.model, test.hex)
threat.hex <- h2o.predict(threat.model, test.hex)
severe_toxic.hex <- h2o.predict(severe_toxic.model, test.hex)
toxic.hex <- h2o.predict(toxic.model , test.hex)
obscene.hex <- h2o.predict(obscene.model, test.hex)


subm <- data.frame(id = test$id)
subm[, "toxic"] <- as.data.table(toxic.hex[,"p1"])
subm[, "severe_toxic"] <- as.data.table(severe_toxic.hex[,"p1"])
subm[, "obscene"] <- as.data.table(obscene.hex[,"p1"])
subm[, "threat"] <- as.data.table(threat.hex[,"p1"])
subm[, "insult"] <- as.data.table(insult.hex[,"p1"])
subm[, "identity_hate"] <- as.data.table(identity_hate.hex[,"p1"])

head(subm)
nrow(subm) #153164
write.csv(subm, '/Users/mauropelucchi/Desktop/Toxic_comments/submission.csv', row.names = FALSE)

#
# Score on Kaggle Leaderboard 0.9043
#
########################################################################
########################################################################
########################################################################


########################################################################
########################################################################
########################################################################
#### Naive Bayes

toxic.model <- h2o.naiveBayes(x = names(comment_all.vecs), y = "toxic",model_id = "toxic_model",
                        training_frame = train.hex, nfolds = 10, 
                        fold_assignment = "Stratified", laplace=0.1)

severe_toxic.model <- h2o.naiveBayes(x = names(comment_all.vecs), y = "severe_toxic",model_id = "severe_toxic_model",
                        training_frame = train.hex, nfolds = 3, 
                        fold_assignment = "Stratified", laplace=0.1)

obscene.model <- h2o.naiveBayes(x = names(comment_all.vecs), y = "obscene",model_id = "obscene_model",
                        training_frame = train.hex, nfolds = 10, 
                        fold_assignment = "Stratified", laplace=0.1)

threat.model <- h2o.naiveBayes(x = names(comment_all.vecs), y = "threat",model_id = "threat_model",
                        training_frame = train.hex, nfolds = 10, 
                        fold_assignment = "Stratified", laplace=0.1)

insult.model <- h2o.naiveBayes(x = names(comment_all.vecs), y = "insult",model_id = "insult_model",
                        training_frame = train.hex, nfolds = 10, 
                        fold_assignment = "Stratified", laplace=0.1)

identity_hate.model <- h2o.naiveBayes(x = names(comment_all.vecs), y = "identity_hate",model_id = "identity_hate_model",
                        training_frame = train.hex, nfolds = 10, 
                        fold_assignment = "Stratified", laplace=0.1)

# Prediction with Naive Bayes
#

identity_hate.hex <- h2o.predict(identity_hate.model, test.hex)
insult.hex <- h2o.predict(insult.model, test.hex)
threat.hex <- h2o.predict(threat.model, test.hex)
severe_toxic.hex <- h2o.predict(severe_toxic.model, test.hex)
toxic.hex <- h2o.predict(toxic.model , test.hex)
obscene.hex <- h2o.predict(obscene.model, test.hex)


subm <- data.frame(id = test$id)
subm[, "toxic"] <- as.data.table(toxic.hex[,"p1"])
subm[, "severe_toxic"] <- as.data.table(severe_toxic.hex[,"p1"])
subm[, "obscene"] <- as.data.table(obscene.hex[,"p1"])
subm[, "threat"] <- as.data.table(threat.hex[,"p1"])
subm[, "insult"] <- as.data.table(insult.hex[,"p1"])
subm[, "identity_hate"] <- as.data.table(identity_hate.hex[,"p1"])

head(subm)
nrow(subm) #153164
write.csv(subm, '/Users/mauropelucchi/Desktop/Toxic_comments/submission.csv', row.names = FALSE)

#
# Score on Kaggle Leaderboard  0.85
#
########################################################################
########################################################################
########################################################################


########################################################################
########################################################################
########################################################################
#### H2O Deep Learning

toxic.model <- h2o.deeplearning(x= names(comment_all.vecs), 
                                  y= "toxic", 
                                  training_frame=train.hex, 
                                  nfolds = 10, 
                                  model_id = "toxic_model",
                                  fold_assignment = "Stratified", 
                                  distribution = "bernoulli",
                                  activation = "RectifierWithDropout",
                                  hidden = c(10,10,10,10),
                                  input_dropout_ratio = 0.2,
                                  l1 = 1e-5,
                                  epochs = 50)

severe_toxic.model <- h2o.deeplearning(x= names(comment_all.vecs), 
                                  y= "severe_toxic", 
                                  training_frame=train.hex, 
                                  nfolds = 10, 
                                  model_id = "severe_toxic_model",
                                  fold_assignment = "Stratified", 
                                  distribution = "bernoulli",
                                  activation = "RectifierWithDropout",
                                  hidden = c(10,10,10,10),
                                  input_dropout_ratio = 0.2,
                                  l1 = 1e-5,
                                  epochs = 50)

obscene.model <- h2o.deeplearning(x= names(comment_all.vecs), 
                                   y= "obscene", 
                                   training_frame=train.hex, 
                                   nfolds = 10, 
                                   model_id = "obscene_model",
                                   fold_assignment = "Stratified", 
                                   distribution = "bernoulli",
                                   activation = "RectifierWithDropout",
                                   hidden = c(10,10,10,10),
                                   input_dropout_ratio = 0.2,
                                   l1 = 1e-5,
                                   epochs = 50)

threat.model <- h2o.deeplearning(x= names(comment_all.vecs), 
                                   y= "threat", 
                                   training_frame=train.hex, 
                                   nfolds = 10, 
                                   model_id = "threat_model",
                                   fold_assignment = "Stratified", 
                                   distribution = "bernoulli",
                                   activation = "RectifierWithDropout",
                                   hidden = c(10,10,10,10),
                                   input_dropout_ratio = 0.2,
                                   l1 = 1e-5,
                                   epochs = 50)

insult.model <- h2o.deeplearning(x= names(comment_all.vecs), 
                                   y= "insult", 
                                   training_frame=train.hex, 
                                   nfolds = 10, 
                                   model_id = "insult_model",
                                   fold_assignment = "Stratified", 
                                   distribution = "bernoulli",
                                   activation = "RectifierWithDropout",
                                   hidden = c(10,10,10,10),
                                   input_dropout_ratio = 0.2,
                                   l1 = 1e-5,
                                   epochs = 50)

identity_hate.model <- h2o.deeplearning(x= names(comment_all.vecs),
                                   y= "identity_hate",
                                   training_frame=train.hex,
                                   nfolds = 10,
                                   model_id = "identity_hate_model1",
                                   fold_assignment = "Stratified",
                                   distribution = "bernoulli",
                                   activation = "RectifierWithDropout",
                                   hidden = c(10,10,10,10),
                                   input_dropout_ratio = 0.2,
                                   l1 = 1e-5,
                                   epochs = 50)

# Prediction with H2O Deep Learning
#

identity_hate.hex <- h2o.predict(identity_hate.model, test.hex)
insult.hex <- h2o.predict(insult.model, test.hex)
threat.hex <- h2o.predict(threat.model, test.hex)
severe_toxic.hex <- h2o.predict(severe_toxic.model, test.hex)
toxic.hex <- h2o.predict(toxic.model , test.hex)
obscene.hex <- h2o.predict(obscene.model, test.hex)


subm <- data.frame(id = test$id)
subm[, "toxic"] <- as.data.table(toxic.hex[,"p1"])
subm[, "severe_toxic"] <- as.data.table(severe_toxic.hex[,"p1"])
subm[, "obscene"] <- as.data.table(obscene.hex[,"p1"])
subm[, "threat"] <- as.data.table(threat.hex[,"p1"])
subm[, "insult"] <- as.data.table(insult.hex[,"p1"])
subm[, "identity_hate"] <- as.data.table(identity_hate.hex[,"p1"])

head(subm)
nrow(subm) #153164
write.csv(subm, '/Users/mauropelucchi/Desktop/Toxic_comments/submission.csv', row.names = FALSE)

#
# Score on Kaggle Leaderboard  0.8753
#
########################################################################
########################################################################
########################################################################