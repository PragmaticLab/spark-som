library(wordVectors)
library(clValid)
library(magrittr)
library(plyr)
setwd("/Users/jason.xie/Downloads/spark-som/sota")

prep_word2vec("/Users/jason.xie/Downloads/spark-som/data/imdb.txt","/Users/jason.xie/Downloads/spark-som/data/imdb_processed.txt", lowercase=T)
model = train_word2vec("/Users/jason.xie/Downloads/spark-som/data/imdb_processed.txt",output="/Users/jason.xie/Downloads/spark-som/model/imdb_R_w2v_model.mod",threads = 3,vectors = 100,window=12,force=TRUE,min_count=15)

sotaCl <- sota(as.matrix(model), 4)
names(sotaCl)
plot(sotaCl)
