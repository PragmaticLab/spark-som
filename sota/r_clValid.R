library(clValid)
library(igraph)
library(plyr)
setwd("/Users/jason.xie/Downloads/spark-som/sota")
data = read.csv("../data/generated_rgb.csv")
colnames(data) = c("r", "g", "b")

data = head(data, 1000)
sotaCl <- sota(as.matrix(data), 4, maxDiversity = 30)
sotaCl
sotaCl$c.tree
#plot(sotaCl)

#t$ID = 
treeDF = data.frame(sotaCl$c.tree)
treeDF[treeDF$ID == 1,]$anc = 1
treeDF = rename(treeDF, c("ID"="id", "anc"="parent"))
treeDF[,c("r", "g", "b")] <- round(treeDF[,c("r", "g", "b")], 0)
treeDF$label = paste(treeDF$r, treeDF$g, treeDF$b, sep="_")
row.names(treeDF) <- treeDF$label

treeGraph = treeDF[,c("parent", "id")]
g <- graph.data.frame(treeGraph)
plot(g, layout = layout.reingold.tilford(g, root=1), vertex.label=row.names(treeGraph),  main="sota graph", vertex.color="white", edge.color="grey", vertex.size=8, vertex.frame.color="yellow")

