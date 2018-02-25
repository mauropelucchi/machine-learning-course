# Kaggle
# Toxic Comment Classification Challenge
# Identify and classify toxic online comments
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# toxic, severe_toxic, obscene ,threat ,insult ,identity_hate

library(data.table)
library(dplyr)
library(stringr)
library(ngram)
library(tokenizers)
library(tidyverse)
library(magrittr)
library(text2vec)
library(glmnet)
library(doParallel)
registerDoParallel(4)

train <- fread("/Users/mauropelucchi/Desktop/Toxic_comments/train.csv", key=c("id"))
test <- fread("/Users/mauropelucchi/Desktop/Toxic_comments/test.csv", key=c("id"))
stopwords.en <- fread("/Users/mauropelucchi/Desktop/Toxic_comments/stopwords-en.txt")
stowwords.custom <- c("put", "far", "bit", "well", "article", "articles", "edit", "edits", "page", "pages",
                      "talk", "page", "editor", "ax", "edu", "subject", "lines", "like", "likes", "line",
                      "uh", "oh", "also", "get", "just", "hi", "hello", "ok", "editing", "edited",
                      "dont", "use", "need", "take", "wikipedia", "give", "say",
                      "look", "one", "make", "come", "see", "said", "now",
                      "wiki", "know", "talk", "read", "hey", "time", "still",
                      "user", "day", "want", "tell", "edit", "even", "ain't", "wow", "image", "jpg", "copyright",
                      "sentence", "wikiproject", "background color", "align", "px", "pixel",
                      "org", "com", "en", "ip", "ip address", "http", "www", "html", "htm",
                      "wikimedia", "https", "httpimg", "url", "urls", "utc", "uhm","username","wikipedia",
                      "what", "which", "who", "whom", "this", "that", "these", "those", 
                      "was", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
                      "doing", "would", "should", "could", "ought", "isn't", "aren't", "wasn't", "weren't", "hasn't",
                      "haven't", "hadn't", "doesn't", "don't", "didn't", "won't", "wouldn't", "shan't", "shouldn't",
                      "can't", "cannot", "couldn't", "mustn't", "let's", "that's", "who's", "what's", "here's",
                      "there's", "when's", "where's", "why's", "how's", "a", "an", "the", "and", "but", "if",
                      "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against",
                      "between", "into", "through", "during", "before", "after", "above", "below", "to", "from",
                      "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
                      "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more",
                      "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
                      "too", "very","articl","ani")

train <- train %>% mutate(filter="train")
test <- test %>% mutate(filter="test")

all_comments <- train %>% bind_rows(test)
nrow(all_comments)

# Create some new features relative to use of punctuation, emotj, ...
all_comments.features <- all_comments %>% 
  
  select(id, comment_text) %>% 
  mutate(length = str_length(comment_text),
         use_cap = str_count(comment_text, "[A-Z]"),
         cap_len = use_cap / length,
         use_lower = str_count(comment_text, "[a-z]"),
         low_len = use_lower / length,
         cap_rate = ifelse(is.null(use_cap / use_lower), 0, use_cap / use_lower),
         cap_odds = ifelse(is.null(cap_len / low_len), 0, cap_len / low_len),
         use_exl = str_count(comment_text, fixed("!")),
         use_space = str_count(comment_text, fixed(" ")),
         use_double_space = str_count(comment_text, fixed("  ")),
         use_quest = str_count(comment_text, fixed("?")),
         use_punt = str_count(comment_text, "[[:punct:]]"),
         use_digit = str_count(comment_text, "[[:digit:]]"),
         digit_len = use_digit / length,
         use_break = str_count(comment_text, fixed("\n")),
         use_word = str_count(comment_text, "\\w+"),
         use_symbol = str_count(comment_text, "&|@|#|\\$|%|\\*|\\^"),
         use_char = str_count(comment_text, "\\W*\\b\\w\\b\\W*"),
         use_i = str_count(comment_text, "(\\bI\\b)|(\\bi\\b)"),
         i_len = use_i / length,
         char_len = use_char / length,
         symbol_len = use_symbol / length,
         use_emotj = str_count(comment_text, "((?::|;|=)(?:-)?(?:\\)|D|P))"),
         cap_emo = use_emotj / length
         ) %>% 
  select(-id) %T>% 
  glimpse()
head(all_comments.features)
nrow(all_comments.features)

# Remove all special chars, clean text and trasform words
all_comments.clean <- all_comments.features %$%
  str_to_lower(comment_text) %>%
  # clear link
  str_replace_all("(f|ht)tp(s?)://\\S+", " ") %>%
  str_replace_all("http\\S+", "") %>%
  str_replace_all("xml\\S+", "") %>%
  # multiple whitspace to one
  str_replace_all("\\s{2}", " ") %>%
  
  # transform short forms
  str_replace_all("what's", "what is ") %>%
  str_replace_all("\\'s", " is ") %>%
  str_replace_all("\\'ve", " have ") %>%
  str_replace_all("can't", "cannot ") %>%
  str_replace_all("n't", " not ") %>%
  str_replace_all("i'm", "i am ") %>%
  str_replace_all("\\'re", " are ") %>%
  str_replace_all("\\'d", " would ") %>%
  str_replace_all("\\'ll", " will ") %>%
  str_replace_all("\\'scuse", " excuse ") %>%
  str_replace_all("pleas", " please ") %>%
  str_replace_all("sourc", " source ") %>%
  str_replace_all("peopl", " people ") %>%
  str_replace_all("remov", " remove ") %>%
  
  # multiple whitspace to one
  str_replace_all("\\s{2}", " ") %>%
  
  # transform shittext
  str_replace_all("(a|e)w+\\b", "") %>%
  str_replace_all("(y)a+\\b", "") %>%
  str_replace_all("(w)w+\\b", "") %>%
  str_replace_all("((a+)|(h+))(a+)((h+)?)\\b", "") %>%
  str_replace_all("((lol)(o?))+\\b", "") %>%
  str_replace_all("n ig ger", " nigger ") %>%
  str_replace_all("s hit", " shit ") %>%
  str_replace_all("g ay", " gay ") %>%
  str_replace_all("f ag got", " faggot ") %>%
  str_replace_all("c ock", " cock ") %>%
  str_replace_all("cu nt", " cunt ") %>%
  str_replace_all("idi ot", " idiot ") %>%
  str_replace_all("f u c k", " fuck ") %>%
  str_replace_all("fu ck", " fuck ") %>%
  str_replace_all("f u ck", " fuck ") %>%
  str_replace_all("c u n t", " cunt ") %>%
  str_replace_all("s u c k", " suck ") %>%
  str_replace_all("c o c k", " cock ") %>%
  str_replace_all("g a y", " gay ") %>%
  str_replace_all("ga y", " gay ") %>%
  str_replace_all("i d i o t", " idiot ") %>%
  str_replace_all("cocksu cking", "cock sucking") %>%
  str_replace_all("du mbfu ck", "dumbfuck") %>%
  str_replace_all("cu nt", "cunt") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\b)", "") %>%
  str_replace_all("(?<=\\w(ck))\\s(?=(ing)\\b)", "") %>%
  str_replace_all("(?<=\\b\\w)\\s(?=\\w\\b)", "") %>%
  str_replace_all("((lol)(o?))+", "") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\b)", "") %>%
  str_replace_all("(?<=\\w(uc))\\s(?=(ing)\\b)", "") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\w)", "") %>%
  str_replace_all("(?<=\\b(fu|su|di|co|li))\\s(?=(k)\\w)", "c") %>%

  # clean nicknames
  str_replace_all("@\\w+", " ") %>%
  
  # clean digit
  str_replace_all("[[:digit:]]", " ") %>%
  
  # remove linebreaks
  str_replace_all("\n", " ") %>%
  
  # remove graphics
  str_replace_all("[^[:graph:]]", " ") %>%
  
  # remove punctuation (if remain...)
  str_replace_all("[[:punct:]]", " ") %>%
                    
  #gsub("\\b\\w{1,2}\\b","", comment_text) %>%
    
  str_replace_all("[^[:alnum:]]", " ") %>%
  
  # remove single char
  str_replace_all("\\W*\\b\\w\\b\\W*", " ")  %>%
  # remove words with len < 2
  str_replace_all("\\b\\w{1,2}\\b", " ")  %>%
  
  # multiple whitspace to one
  str_replace_all("\\s{2}", " ") %>%
  str_replace_all("\\s+", " ") %>%
  
  itoken(tokenizer = tokenize_word_stems)

# Calculate stopwords and rarewords
stop_words_chr <- c(stopwords("en")) %>% c(stopwords.en$C1) %>% c(stowwords.custom)

# Vectorizer
vectorizer.dict <- create_vocabulary(all_comments.clean, ngram = c(1L, 1L), stopwords = stop_words_chr) %>%
  prune_vocabulary(term_count_min = 4, doc_proportion_max = 0.3, vocab_term_max = 10000)

print(vectorizer.dict)

vectorizer <- vectorizer.dict %>% vocab_vectorizer()
all_comments.dtm <- create_dtm(all_comments.clean, vectorizer) %>% normalize(norm = "l2")

names(all_comments.features)

# Preparing data for glmnet
all_comments.matrix <- all_comments.features %>% 
  select(-comment_text) %>% 
  sparse.model.matrix(~ . - 1, .) %>% 
  cbind(all_comments.dtm)

# Prepare train and test set
names(all_comments)
train.set <- all_comments[, "filter"] == "train"
test.set <- all_comments[, "filter"] == "test"
train.matrix <- all_comments.matrix[train.set,]
test.matrix <- all_comments.matrix[test.set,]
rm(all_comments.features, all_comments.clean, vectorizer, all_comments.dtm); gc()

subm <- data.frame(id = test$id)


# Training glmnet & predict toxicity
for (target in c("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")) {
  cat("\nTrain -->", target, "...")
  y <- factor(train[[target]])
  glm.model <- cv.glmnet(train.matrix, y, alpha = 0, family = "binomial", type.measure = "auc",
                     parallel = T, standardize = T, nfolds = 10, nlambda = 50)
  cat(" AUC:", max(glm.model$cvm))
  subm[[target]] <- predict(glm.model, test.matrix, type = "response", s = "lambda.min")
}

head(subm)
nrow(subm) #153164
write.csv(subm, '/Users/mauropelucchi/Desktop/Toxic_comments/submission.csv', row.names = FALSE)


  

