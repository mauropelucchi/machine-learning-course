# Kaggle
# Toxic Comment Classification Challenge
# Identify and classify toxic online comments
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# toxic, severe_toxic, obscene ,threat ,insult ,identity_hate
#
# A novel story from exploration analysis of toxic comments
#
#

library(data.table)
library(dplyr)
library(h2o)
library(stringr)
library(ngram)
packageVersion("h2o")

#h2o.init(ip = "localhost", port = 4444, nthreads = 15, max_mem_size = "4g", strict_version_check = FALSE)
h2o.init(ip = "localhost", port = 4444, nthreads = -1, max_mem_size = "8g", strict_version_check = FALSE)

train <- fread("/Users/mauropelucchi/Desktop/Toxic_comments/train.csv", key=c("id"))
test <- fread("/Users/mauropelucchi/Desktop/Toxic_comments/test.csv", key=c("id"))
stopwords.en <- fread("/Users/mauropelucchi/Desktop/Toxic_comments/stopwords-en.txt")

train <- train %>% mutate(filter="train")
test <- test %>% mutate(filter="test")

all_comments <- train %>% bind_rows(test)
nrow(all_comments)



## Count frequency for each class (on train)
nrow(train)
library(ggplot2)
library(tidyr)
names(tmp)
tmp <- train %>% summarise(toxic=sum(toxic), 
                    severe_toxic=sum(severe_toxic), 
                    obscene=sum(obscene) ,
                    threat=sum(threat) ,
                    insult=sum(insult) ,
                    identity_hate=sum(identity_hate),
                    total=n()) %>%
  mutate(ok=total-toxic-severe_toxic-obscene-threat-insult-identity_hate) %>%
  select(ok,toxic,obscene,insult,severe_toxic,identity_hate,threat)
# transpose all but the first column (name)
df.freq <- as.data.frame(t(tmp[,]))
colnames(df.freq ) <- t(tmp[0, ])
df.freq$toxicity <- factor(row.names(df.freq))
colnames(df.freq)[1] <- "n"

df.freq %>%
  ggplot(aes(x=reorder(toxicity,-n), y=n, fill=toxicity)) +
  geom_bar(stat="identity") +
  theme(axis.text.x=element_text(angle=90, hjust=1), axis.title.x = element_blank())# 



# Remove all special chars
all_comments <- all_comments %>% mutate(clean_text=str_replace_all(comment_text, "[^[:alnum:]]", " "))
head(all_comments)
# Go to H2O
comments.hex <- as.h2o(select(all_comments, c("id", "clean_text", "filter", "toxic", "severe_toxic", "obscene","threat" ,"insult" ,"identity_hate")), destination_frame = "all_comments.hex", col.types=c("String"))


# Tokenize, lowercase, remove text with len less than 3 chars
tokenize <- function(text) {
  tokenized <- h2o.tokenize(text, "\\\\W+")
  tokenized.lower <- h2o.tolower(tokenized)
  # remove (less than 2 chars)
  tokenized.lengths <- h2o.nchar(tokenized.lower)
  tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths > 2,]
  # remove words that contain numbers
  tokenized.words <- tokenized.filtered[h2o.grep("[0-9]", tokenized.filtered, invert = TRUE, output.logical = TRUE),]
}


# Get a list of terms and create stop_words and rare_words lists
words.hex <- tokenize(comments.hex$clean_text)
h2o.head(words.hex)
words <- as.data.frame(words.hex)
words <- words %>% group_by(C1) %>% mutate(wc=n())
tmp <- words %>% tally(wc) %>% top_n(200)
head(tmp)
tmp <- words %>% tally(wc) %>% top_n(-200)
tmp
stop_words <- words %>% tally(wc) %>% top_n(200) %>% select(C1) %>% ungroup()
rare_words <- words %>% filter(wc < 10) %>% select(C1) %>% ungroup()
rm(words, tmp)
h2o.rm(words.hex)
#
stop_words_chr <- c(stop_words$C1) %>% union(stopwords.en$C1) %>% c('wikipedia','username')
rare_words_chr <- c(rare_words$C1)


# Tokenize, lowercase, remove text with len less than 3 chars
clean_token <- function(text, STOP_WORDS, RARE_WORDS) {
  tokenized <- h2o.tokenize(text, "\\\\W+")
  tokenized.lower <- h2o.tolower(tokenized)
  # remove (less than 2 chars)
  tokenized.lengths <- h2o.nchar(tokenized.lower)
  tokenized.filtered <- tokenized.lower[is.na(tokenized.lengths) || tokenized.lengths > 2,]
  # remove words that contain numbers
  tokenized.words <- tokenized.filtered[h2o.grep("[0-9]", tokenized.filtered, invert = TRUE, output.logical = TRUE),]
  tokenized.words <- tokenized.words[is.na(tokenized.words) || (! tokenized.words %in% STOP_WORDS),]
  tokenized.words <- tokenized.words[is.na(tokenized.words) || (! tokenized.words %in% RARE_WORDS),]
  tokenized.words[!is.na(tokenized.words), ]
}

## Explore Common words
words.hex <- clean_token(comments.hex$clean_text, stop_words_chr, rare_words_chr)
h2o.head(words.hex)
nrow(words.hex) #6128533
words <- as.data.table(words.hex)
words <- words %>% group_by(C1) %>% mutate(wc=n())
top.words <- words %>% tally(wc) %>% top_n(10) %>% ungroup()
head(top.words, 10)
names(top.words)[1] <- "word"
names(top.words)[2] <- "n"


top.words %>%
  ggplot(aes(x=reorder(word,-n), y=n, fill=word)) +
  geom_bar(stat="identity") +
  theme(axis.text.x=element_text(angle=90, hjust=1), axis.title.x = element_blank())# 



## Explore Toxic comment
toxic.set <- comments.hex[, "toxic"] == 1
toxic.hex <- comments.hex[toxic.set, ]
toxic.words.hex <- clean_token(toxic.hex$clean_text, stop_words_chr, rare_words_chr)
h2o.head(toxic.words.hex)
nrow(toxic.words.hex) #251819
toxic.words <- as.data.table(toxic.words.hex)
toxic.words <- toxic.words %>% group_by(C1) %>% mutate(wc=n())
top.toxic_words <- toxic.words %>% tally(wc) %>% top_n(10) %>% ungroup()
head(top.toxic_words, 10)
names(top.toxic_words)[1] <- "word"
names(top.toxic_words)[2] <- "n"


top.toxic_words %>%
  ggplot(aes(x=reorder(word,-n), y=n, fill=word)) +
  geom_bar(stat="identity") +
  theme(axis.text.x=element_text(angle=90, hjust=1), axis.title.x = element_blank())# 





## Explore Severe toxic comment
severe_toxic.set <- comments.hex[, "severe_toxic"] == 1
severe_toxic.hex <- comments.hex[severe_toxic.set, ]
severe_toxic.words.hex <- clean_token(severe_toxic.hex$clean_text, stop_words_chr, rare_words_chr)
h2o.head(severe_toxic.words.hex)
nrow(severe_toxic.words.hex) #46267
severe_toxic.words <- as.data.table(severe_toxic.words.hex)
severe_toxic.words <- severe_toxic.words %>% group_by(C1) %>% mutate(wc=n())
top.severe_toxic_words <- severe_toxic.words %>% tally(wc) %>% top_n(10) %>% ungroup()
head(top.severe_toxic_words, 10)
names(top.severe_toxic_words)[1] <- "word"
names(top.severe_toxic_words)[2] <- "n"


top.severe_toxic_words %>%
  ggplot(aes(x=reorder(word,-n), y=n, fill=word)) +
  geom_bar(stat="identity") +
  theme(axis.text.x=element_text(angle=90, hjust=1), axis.title.x = element_blank())# 




## Explore Obscene
obscene.set <- comments.hex[, "obscene"] == 1
obscene.hex <- comments.hex[obscene.set, ]
obscene.words.hex <- clean_token(obscene.hex$clean_text, stop_words_chr, rare_words_chr)
h2o.head(obscene.words.hex)
nrow(obscene.words.hex) #137138
obscene.words <- as.data.table(obscene.words.hex)
obscene.words <- obscene.words %>% group_by(C1) %>% mutate(wc=n())
top.obscene_words <- obscene.words %>% tally(wc) %>% top_n(10) %>% ungroup()
head(top.obscene_words, 10)
names(top.obscene_words)[1] <- "word"
names(top.obscene_words)[2] <- "n"


top.obscene_words %>%
  ggplot(aes(x=reorder(word,-n), y=n, fill=word)) +
  geom_bar(stat="identity") +
  theme(axis.text.x=element_text(angle=90, hjust=1), axis.title.x = element_blank())# 




## Explore threat
threat.set <- comments.hex[, "threat"] == 1
threat.hex <- comments.hex[threat.set, ]
threat.words.hex <- clean_token(threat.hex$clean_text, stop_words_chr, rare_words_chr)
h2o.head(threat.words.hex)
nrow(threat.words.hex) #9219
threat.words <- as.data.table(threat.words.hex)
threat.words <- threat.words %>% group_by(C1) %>% mutate(wc=n())
top.threat_words <- threat.words %>% tally(wc) %>% top_n(10) %>% ungroup()
head(top.threat_words, 10)
names(top.threat_words)[1] <- "word"
names(top.threat_words)[2] <- "n"


top.threat_words %>%
  ggplot(aes(x=reorder(word,-n), y=n, fill=word)) +
  geom_bar(stat="identity") +
  theme(axis.text.x=element_text(angle=90, hjust=1), axis.title.x = element_blank())# 


## Explore insult
insult.set <- comments.hex[, "insult"] == 1
insult.hex <- comments.hex[insult.set, ]
insult.words.hex <- clean_token(insult.hex$clean_text, stop_words_chr, rare_words_chr)
h2o.head(insult.words.hex)
nrow(insult.words.hex) #9219
insult.words <- as.data.table(insult.words.hex)
insult.words <- insult.words %>% group_by(C1) %>% mutate(wc=n())
top.insult_words <- insult.words %>% tally(wc) %>% top_n(10) %>% ungroup()
head(top.insult_words, 10)
names(top.insult_words)[1] <- "word"
names(top.insult_words)[2] <- "n"


top.insult_words %>%
  ggplot(aes(x=reorder(word,-n), y=n, fill=word)) +
  geom_bar(stat="identity") +
  theme(axis.text.x=element_text(angle=90, hjust=1), axis.title.x = element_blank())# 



## Explore identity_hate
identity_hate.set <- comments.hex[, "identity_hate"] == 1
identity_hate.hex <- comments.hex[identity_hate.set, ]
identity_hate.words.hex <- clean_token(identity_hate.hex$clean_text, stop_words_chr, rare_words_chr)
h2o.head(identity_hate.words.hex)
nrow(identity_hate.words.hex) #27081
identity_hate.words <- as.data.table(identity_hate.words.hex)
identity_hate.words <- identity_hate.words %>% group_by(C1) %>% mutate(wc=n())
top.identity_hate_words <- identity_hate.words %>% tally(wc) %>% top_n(10) %>% ungroup()
head(top.identity_hate_words, 10)
names(top.identity_hate_words)[1] <- "word"
names(top.identity_hate_words)[2] <- "n"


top.identity_hate_words %>%
  ggplot(aes(x=reorder(word,-n), y=n, fill=word)) +
  geom_bar(stat="identity") +
  theme(axis.text.x=element_text(angle=90, hjust=1), axis.title.x = element_blank())# 




####################################################################################
# Build word2vec model
vectors <- 10
w2v.model <- h2o.word2vec(words.hex
                          , model_id = "w2v_model"
                          , vec_size = vectors
                          , min_word_freq = 10
                          , window_size = 5
                          , init_learning_rate = 0.025
                          , sent_sample_rate = 0
                          , epochs = 18)


##### Check - find synonyms for the word 'water',mexicans','hardcore','metallica','hot','idiot'
print(h2o.findSynonyms(w2v.model, "water", count = 5))
print(h2o.findSynonyms(w2v.model, "hardcore", count = 5))
print(h2o.findSynonyms(w2v.model, "metallica", count = 5))
print(h2o.findSynonyms(w2v.model, "hot", count = 5))
print(h2o.findSynonyms(w2v.model, "idiot", count = 5))
print(h2o.findSynonyms(w2v.model, "mexicans", count = 5))


