# ADA Project 2021: Event Based Trading Algorithm
Project on Event Based Trading for the Course Advanced Data Analytics - HEC Lausanne - Spring 2021

![alt text](https://camo.githubusercontent.com/c327657381291ed9f2e8866cb96ac4861431d9c244b7b14dcf4e1470cbf632da/68747470733a2f2f75706c6f61642e77696b696d656469612e6f72672f77696b6970656469612f636f6d6d6f6e732f7468756d622f612f61332f4845435f4c617573616e6e655f6c6f676f2e7376672f32393370782d4845435f4c617573616e6e655f6c6f676f2e7376672e706e67)

## Project Content

Our project will evaluate if it is possible to predict the direction of asset’s price movement using data from new information available on the internet. This project will be interesting in the context of Efficient Market Hypothesis and the latest market turmoil regarding r/wallstreetbets and Gamestop. To implement this, we will try to predict the price movement of the S&P500 index, TSLA and BitCoin, using the posts on Reddit, particularly under the r/worldnews, r/investing, r/finance, r/wallstreetbets and r/CryptoCurrency subreddit. To do so, we will determine the average daily sentiment of each subreddit using [Textblob](https://github.com/sloria/TextBlob ) and [Vader](https://github.com/cjhutto/vaderSentiment) disctinctively, as none of them are perfectly suited for Reddit contents (VADER is more appropriate for Twitter for example). To install the libraries:

* Vader: pip install vaderSentiment
* Textblob: pip install -U textblob

## Data 
We collected the top 25 commented posts under the aforementioned subreddits from January 2014 until May 2021 with the Reddit API using [praw](https://github.com/praw-dev/praw) and [psaw](https://github.com/dmarx/psaw) to collect the top posts each day. To install the libraries: 

* Praw:  pip install praw
* Psaw:  pip install psaw

To use the Reddit API, we created a reddit "application" to obtain the credentials of collecting the posts:

* Create an account: https://www.reddit.com/wiki/api

Using the non-official [Yahoo Finance API](https://github.com/ranaroussi/yfinance), since Yahoo finance decommissioned their historical data API, we collected the prices of the following assets/index from January 2014 until May 2021:

* S&P500 Index (Ticker: ^GSPC)
* Tesla Inc (Ticker: TSLA)
* BitCoin (Ticker: BTC-USD)
* VIX Index (Ticker: ^VIX)
* 10y T-Bills (Ticker: ^TNX)

## Model
To predict the price movement (up:1 , down:0), we used the following classification algortithms:
* K-Nearest Neighbours
* Support Vector Machine
* Random Forest
* Adapative Boosting
* Naive Bayes
* Logistic Regression
* Long-Short Term Memory (LSTM)
* Deep Neural Networks (DNN)

For all our models, except the LSTM and DNN, we applied hyperparameters tuning using grid search and random search. The best model model was selected basey on its accuracy.

## Output
To present our result, we will compare a hold-only position of the corresponding asset to a long/short position based on our predicitons (i.e. long if we predict a price increase, short if we except a price decrease).

## Authors

* Sebastien Gorgoni
* Liam Svoboda

## Acknowledgments

* Prof. [Simon Sheidegger](https://sites.google.com/site/simonscheidegger/home)
* [Antoine Didisheim](https://www.antoinedidisheim.com)
* Everyone who shared publicly their codes on Github :+1:
* Every beloved [Reddit Mods](https://www.youtube.com/watch?v=5DFCOh0mtYs&t=91s) who moderate our selected subreddits :heart:

