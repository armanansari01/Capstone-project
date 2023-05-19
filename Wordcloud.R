capdata <- read.csv(file.choose())
str(capdata)
View(capdata)
text_index <- paste(capdata$Situation_sarcasm , 
                    capdata$perceive_sarcasm , 
                    capdata$impact_on_relationship , capdata$person_response , 
                    capdata$Experience_depression , capdata$Views_on_depression, 
                    colorPalette = "RdBu")
library(wordcloud)
wordcloud(text_index)

library(tm)

library(syuzhet)

reviews <- c(capdata$Views_on_depression,capdata$Experience_depression,
             capdata$perceive_sarcasm,capdata$Situation_sarcasm)

corpus <- Corpus(VectorSource(iconv(reviews)))
inspect(corpus[1:5])

corpus <- tm_map(corpus, tolower)
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords)
inspect(corpus[1:5])

final_reviews <- corpus

dtm <- TermDocumentMatrix(final_reviews)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 10)

set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"),scale=c(3,0.3))

sd <- get_nrc_sentiment(char_v= d$word)
sd[1:10,]

sdupdated <- colSums(sd[ ,1:10])
sdupdated

sd$Score <- sd$positive - sd$negative
sd$Score

library(ggplot2)

transform(table(sdupdated))

sd_df <- as.data.frame(sdupdated)
Frequency<- sd_df$sdupdated

Sentiment <- c("anger","anticipation","disgust","fear","joy","sadness",
               "surprise","trust","negative","positive")

df <- data.frame(Sentiment,Frequency)

ggplot(df, aes(x=Sentiment, y=Frequency)) +
             geom_bar(stat = "identity",color="blue",fill=rgb(0.1,0.4,0.5,0.7))
