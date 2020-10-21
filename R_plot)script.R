data <- read.csv('psd_channel_data_single_channel.csv')

X <- data[,2:24]
y <- data[,25]

bor.results <- Boruta(X,y,maxRuns = 101,doTrace = 0)

{r borutaplot, echo=FALSE , fig.width=9,fig.height=7}
plot(bor.results,sort = FALSE)