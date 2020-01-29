require(quanteda)
require(readxl)
require(gsubfn)
require(tm)
require(stm)
require(topicmodels)
require(tidyr)
require(dplyr)
require(gsubfn)
require(stopwords)
library(dplyr)

## Load GenJ
df <- read.delim("general_journals_sections.csv", header=T, fill=T, sep=',')

## Turn sections into composites
df <- df %>%
  group_by(ID_jstor) %>%
  summarise(text = paste0(text, collapse = " "), author_genders = first(author_genders), year = first(year))

## Processing to DFM
HistCorpus <- corpus(df$text, docvars = df[,-4])
HistToks <- tokens(HistCorpus, remove_punct = T,)
HistDFM <-  dfm(HistCorpus,
                tolower = T,
                stem = T,
                remove = stopwords("English"),) # process
HistDFM <- dfm_trim(HistDFM, min_docfreq = 2, docfreq_type = "count") #ngrams happening >= 2x

## Remove words with less than 3 characters
HistDFM <- dfm_keep(HistDFM, "\\w{3,}", valuetype = "regex")

# Word frequencies, useful for confirming stopwords
freq.df <- textstat_frequency(HistDFM)
freq.df$prop <- freq.df$frequency/ndoc(HistDFM)
write.csv(freq.df, "word_frequencies_Gen_Sec_Sherl.csv")

# Process for STM
data <-df
y<-textProcessor(data$text,
                 metadata = data,
                 lowercase = TRUE,
                 removestopwords = TRUE,
                 removenumbers = TRUE,
                 removepunctuation = FALSE,
                 stem = TRUE)
pre1<-prepDocuments(y$documents, y$vocab, y$meta, lower.thresh = 1)

# Find K value, generate graph
set.seed(2024)
kresult3<-searchK(pre1$documents, pre1$vocab, K = c(20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140),  prevalence=~author_genders + s(year), data= pre1$meta, cores=1)
plot(kresult3)

## 100 looks good, we'll use that
HistDFM.stm <- dfm_subset(HistDFM, df$author_genders != "unknown")
HistDFM.stm <- dfm_subset(HistDFM, df$author_genders != "mixed")
HistDFM.stm <- convert(HistDFM.stm, to = "stm")
HistDFM.stm$meta$author_genders <- as.factor(HistDFM.stm$meta$author_genders)
HistSTM <- stm(documents = HistDFM.stm$documents,
               vocab = HistDFM.stm$vocab,
               data = HistDFM.stm$meta,
               K = 100,)

## Topic vars generation
theta.df <- data.frame(HistSTM$theta)
colnames(theta.df) <- paste("topic", 1:K, sep = ".")
theta.df <- cbind(HistDFM.stm$meta, theta.df)
write.csv(theta.df, "doc_with_topiccovars_stm_GENJ_100_full_Sherl.csv")

## Topic terms
words =  labelTopics(HistSTM, n=20)
probs = data.frame(words$prob)
colnames(probs)  = 1:20
probs$prob =  apply( probs[ , 1:20 ] , 1 , paste , collapse = ", " )
frexs = data.frame(words$frex)
colnames(frexs)  = 1:7
frexs$frex =  apply( frexs[ , 1:20 ] , 1 , paste , collapse = ", " )
probs$name = 1:K
frexs$name = 1:K
words = merge(probs[,c('name', 'prob')],
              frexs[,c('name', 'frex')],
              by = 'name')

write.csv(words, "topic_words_stm_GENJ_100_full.csv")