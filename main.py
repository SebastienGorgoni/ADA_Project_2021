"""
ADVANCED DATA ANALYSIS 2021

EVENT BASED TRADING ALGORITHM: A REDDIT CASE

Authors: Sebastien Gorgoni & Liam Svoboba

File Name: main.py

This is the main file of the project called "EVENT BASED TRADING ALGORITHM: A REDDIT CASE". It is divided into X part:
    
    1) Extract Top Commented Reddit Post
    2) Create the Average Daily Sentiment of each Subreddits
    3) Extract the Financial Time Series
    4) Description of Data Set
    5) Group the Financial Data with Reddit Post Sentiment (Preparation for the Predictions)
    6) Create a Function that Run all Predictions 
    7) Evalutation of our Predictions (Long/Short Positions) Compared to Long-Hold Only 
    8) Run all predictions

"""

# Import all External Libraires
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import os
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob 
import tensorflow as tf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="darkgrid")

np.random.seed(42)
tf.random.set_seed(42)

# Set the Working Directory
os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 4.2/Advanced Data Analytics/ADA_Project")
print("Current working directory: {0}".format(os.getcwd()))

# Import the models and reddit download from prediction_models.py and reddit_scratch_private.py
from prediction_models import logistic, gradboost, adaboost, svm, randomforest, knn, sgd, mlp, naivebayes, ann, lstm
from reddit_scratch_private import download_reddit

# =============================================================================
# PART 1: Extract Top Commented Reddit Post from 01-01-15 to 01-05-21
# =============================================================================

############## If CSV file does not exist yet ##############

"""
start = dt.datetime(2014, 1, 1).date()
end = dt.datetime(2021, 5, 1).date()
delta = end - start
index = pd.date_range(start+dt.timedelta(1), periods=(delta).days, freq='D')

df_worldnews = download_reddit(index, 25, 'worldnews', 'test_ada_reddit_worldnews.csv')
df_finance = download_reddit(index, 25, 'finance', 'test_ada_reddit_finance.csv')
df_crypto = download_reddit(index, 25, 'CryptoCurrency', 'test_ada_reddit_CryptoCurrency.csv')
df_wsb = download_reddit(index, 25, 'wallstreetbets', 'test_ada_reddit_wsb.csv')
df_investing = download_reddit(index, 25, 'investing', 'test_ada_reddit_investing.csv') 
"""

############## If CSV file already exist ##############

os.chdir("/Users/sebastiengorgoni")

df_worldnews = pd.read_csv('test_ada_reddit_worldnews.csv')
df_worldnews.set_index('Date', inplace=True)
df_worldnews.index =  pd.to_datetime(df_worldnews.index, format='%Y/%m/%d')

df_finance = pd.read_csv('test_ada_reddit_finance.csv')
df_finance.set_index('Date', inplace=True)
df_finance.index =  pd.to_datetime(df_finance.index, format='%Y-%m-%d')

df_crypto = pd.read_csv('test_ada_reddit_CryptoCurrency.csv')
df_crypto.set_index('Date', inplace=True)
df_crypto.index =  pd.to_datetime(df_crypto.index, format='%Y-%m-%d')

df_wsb = pd.read_csv('test_ada_reddit_wsb.csv')
df_wsb.set_index('Date', inplace=True)
df_wsb.index =  pd.to_datetime(df_wsb.index, format='%Y-%m-%d')

df_investing = pd.read_csv('test_ada_reddit_investing.csv')
df_investing.set_index('Date', inplace=True)
df_investing.index =  pd.to_datetime(df_investing.index, format='%Y-%m-%d')

os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 4.2/Advanced Data Analytics/ADA_Project")

###########

# =============================================================================
# PART 2: Create the Average Daily Sentiment of each Sub-reddit
# =============================================================================

"""Create average daily sentiment of each subreddit using Textblob"""
def textblob(df):
    temp = pd.DataFrame()
    for i in range(1, df.shape[1]+1):
        #temp[f'Top{i}'] = df[f'Top{i}'].apply(lambda headline: TextBlob(headline).sentiment.polarity)
        temp[f'Top{i}'] = df[f'Top{i}'].apply(lambda x: np.nan if pd.isnull(x) else TextBlob(x).sentiment.polarity)
    return temp.mean(axis=1)

df_worldnews_t = pd.DataFrame({'Sentiment r/worldnews': textblob(df_worldnews)})
df_finance_t = pd.DataFrame({'Sentiment r/finance': textblob(df_finance)})
df_crypto_t = pd.DataFrame({'Sentiment r/CryptoCurrency': textblob(df_crypto)})
df_wsb_t = pd.DataFrame({'Sentiment r/wallstreetbets': textblob(df_wsb)})
df_investing_t = pd.DataFrame({'Sentiment r/investing': textblob(df_investing)})

"""Create average daily sentiment of each subreddit using Vader"""
def vader(df):
    analyzer = SentimentIntensityAnalyzer()
    temp = df.copy()
    for i in range(1, df.shape[1]+1):
        #temp[f'Top{i}'] = [analyzer.polarity_scores(x)['compound'] for x in df[f'Top{i}']]
        temp[f'Top{i}'] = df[f'Top{i}'].apply(lambda x: np.nan if pd.isnull(x) else analyzer.polarity_scores(x)['compound'])
    return temp.mean(axis=1)

df_worldnews_v = pd.DataFrame({'Sentiment r/worldnews': vader(df_worldnews)}, index = df_worldnews.index)
df_finance_v = pd.DataFrame({'Sentiment r/finance': vader(df_finance)}, index = df_finance.index)
df_crypto_v = pd.DataFrame({'Sentiment r/CryptoCurrency': vader(df_crypto)}, index = df_crypto.index)
df_wsb_v = pd.DataFrame({'Sentiment r/wallstreetbets': vader(df_wsb)}, index = df_wsb.index)
df_investing_v = pd.DataFrame({'Sentiment r/investing': vader(df_investing)}, index = df_investing.index)

"""Determine Whether Bitcoin or Tesla/Elon Musk Was Mentionned In a Given Day"""
merged = pd.concat([df_worldnews, df_finance, df_crypto, df_wsb, df_investing], axis=1)
merged = merged.where(pd.notnull(merged), str(None))

tesla_headlines = []
for i in range(0, merged.shape[0]):
    temp = []
    for j in range(0, merged.shape[1]):
        temp.append(len(re.findall('Elon|elon|Musk|musk|tesla|Tesla|TESLA|TSLA|tsla|$TSLA', merged.iloc[i, j])))
    tesla_headlines.append(np.sum(temp))

btc_headlines = []
for i in range(0, merged.shape[0]):
    temp = []
    for j in range(0, merged.shape[1]):
        temp.append(len(re.findall('Bitcoin|BitCoin|bitcoin|BTC|btc|$BTC', merged.iloc[i, j])))
    btc_headlines.append(np.sum(temp))

freq_headlines = pd.DataFrame({'Tesla Freq.': tesla_headlines, 'BTC Freq.': btc_headlines}, index=merged.index)

for i in freq_headlines.columns:
    freq_headlines.loc[freq_headlines[i] >= 1, i] = 1
    freq_headlines.loc[freq_headlines[i] < 1, i] = 0

# =============================================================================
# PART 3: Extract the Financial Time Series
# =============================================================================

start_fin = dt.datetime(2014, 1, 1).date()
end_fin = dt.datetime(2021, 5, 1).date()

#Get Historical SP500 Value in USD
sp500 = yf.download(tickers = '^GSPC', start=start_fin, end=end_fin)
sp500['SP500 Returns'] = ((sp500['Close']/sp500['Close'].shift(1))-1).dropna(how='any')
sp500 = sp500.dropna()
sp500['SP500 Jump'] = 0
sp500.loc[sp500['Close'] > sp500['Open'], 'SP500 Jump'] = 1

#Get Historical Tesla Inc. Price in USD
tesla = yf.download(tickers = "TSLA", start=start_fin, end=end_fin)
tesla['TSLA Returns'] = ((tesla['Close']/tesla['Close'].shift(1))-1).dropna(how='any')
tesla = tesla.dropna()
tesla['TSLA Jump'] = 0
tesla.loc[tesla['Close'] > tesla['Open'], 'TSLA Jump'] = 1

#Get Historical BitCoin. Price in USD
btc = yf.download(tickers = "BTC-USD", start=start_fin, end=end_fin)
btc['BTC Returns'] = ((btc['Close']/btc['Close'].shift(1))-1).dropna(how='any')
btc = btc.dropna()
btc['BTC Jump'] = 0
btc.loc[btc['Close'] > btc['Open'], 'BTC Jump'] = 1

#Get Historical VIX index in USD
vix = yf.download(tickers = "^VIX", start=start_fin, end=end_fin)
vix['VIX Returns'] = ((vix['Close']/vix['Close'].shift(1))-1).dropna(how='any')
vix = vix.dropna()
vix['VIX Jump'] = 0
vix.loc[vix['Close'] > vix['Open'], 'VIX Jump'] = 1
vix.rename(columns={'Close': 'VIX Index'}, inplace=True)

#Get Historical 10y Tbill Rate
tbill = yf.download(tickers = "^TNX", start=start_fin, end=end_fin)
tbill['TBill Jump'] = 0
tbill .loc[tbill['Close'] > tbill ['Open'], 'TBill Jump'] = 1
tbill.rename(columns={'Close': '10y Tbill'}, inplace=True)
tbill['10y Tbill'] = tbill['10y Tbill']/100

# =============================================================================
# PART 4: Description of Data Set
# =============================================================================

# Create a function to compute the cumulative returns 
def cum_prod(returns):
    return (returns + 1).cumprod()*100

if not os.path.isdir('Plot_descr'):
    os.makedirs('Plot_descr')

# Plot the Correlation Matrix of each Financial time series we collected
plt.figure(figsize=(10,7))
corr = pd.DataFrame(pd.concat([sp500['SP500 Returns'], tesla['TSLA Returns'], btc['BTC Returns'], vix['VIX Returns'], tbill['10y Tbill']], axis=1)).corr()
sns.heatmap(corr, annot=True)
plt.savefig('Plot_descr/corr_descr.png')
plt.show()
plt.close()

# Plot the cumulative returns of SP500 index, TSLA and BTCUS
plt.figure(figsize=(13,7))
plt.subplot(131)
plt.plot(cum_prod(sp500['SP500 Returns']), 'r', label='S&P500 Index', linewidth=2)
plt.title('Cumulative Performance')
plt.legend(loc='upper left', frameon=True)
plt.subplot(132)
plt.plot(cum_prod(tesla['TSLA Returns']), 'g', label='Tesla Inc.', linewidth=2)
plt.title('Cumulative Performance')
plt.legend(loc='upper left', frameon=True)
plt.subplot(133)
plt.plot(cum_prod(btc['BTC Returns']), 'orange', label='BTC-USD', linewidth=2)
plt.title('Cumulative Performance')
plt.legend(loc='upper left', frameon=True)
plt.savefig('Plot_descr/cumul_perf_descr.png')
plt.show()
plt.close()

# Plot the average number of time in which the price increased/decreased of SP500 index, TSLA and BTCUS
label=['Decreased (0)', 'Increased (1)']
colors = ['r', 'b']
plt.figure(figsize=(10,4))
plt.subplot(131)
plt.bar(label, [sp500['SP500 Jump'].loc[sp500['SP500 Jump']==0].count() / sp500.shape[0],sp500['SP500 Jump'].loc[sp500['SP500 Jump']==1].count()/sp500.shape[0]], color=colors)
plt.title('S&P500 Index', fontsize=14)
plt.subplot(132)
plt.bar(label, [tesla['TSLA Jump'].loc[tesla['TSLA Jump']==0].count() / tesla.shape[0], tesla['TSLA Jump'].loc[tesla['TSLA Jump']==1].count()/tesla.shape[0]], color=colors)
plt.title('Tesla Inc.', fontsize=14)
plt.subplot(133)
plt.bar(label, [btc['BTC Jump'].loc[btc['BTC Jump']==0].count() / btc.shape[0], btc['BTC Jump'].loc[btc['BTC Jump']==1].count()/btc.shape[0]], color=colors)
plt.title('BTC-USD', fontsize=14)
plt.savefig('Plot_descr/jump_descr.png')
plt.show()
plt.close()

# Plot the evolution of VIX and 10y Tbill
plt.figure(figsize=(13,7))
plt.subplot(121)
plt.plot(tbill['10y Tbill'], 'b')
plt.title('10y TBill')
plt.subplot(122)
plt.plot(vix['VIX Index'], 'r')
plt.title('VIX Index')
plt.savefig('Plot_descr/VIX_tbill_descr.png')
plt.show()
plt.close()

# =============================================================================
# PART 5: Group the Price DataFrame with Reddit Post sentiment
# =============================================================================

"""Get the Returns of each Asset from the previous Day"""
sp500_lag1 = pd.DataFrame({'Lag SP500 Return': sp500['SP500 Returns'].shift(1, axis = 0).dropna()})
tesla_lag1 = pd.DataFrame({'Lag TSLA Return': tesla['TSLA Returns'].shift(1, axis = 0).dropna()})
btc_lag1 = pd.DataFrame({'Lag BTC Return': btc['BTC Returns'].shift(1, axis = 0).dropna()})

"""Normalize the Data"""
df_worldnews_t = (df_worldnews_t - df_worldnews_t.mean(axis=0))/df_worldnews_t.std(axis=0)
df_finance_t = (df_finance_t - df_finance_t.mean(axis=0))/df_finance_t.std(axis=0)
df_crypto_t = (df_crypto_t - df_crypto_t.mean(axis=0))/df_crypto_t.std(axis=0)
df_wsb_t = (df_wsb_t - df_wsb_t.mean(axis=0))/df_wsb_t.std(axis=0)
df_investing_t = (df_investing_t - df_investing_t.mean(axis=0))/df_investing_t.std(axis=0)

df_worldnews_v = (df_worldnews_v - df_worldnews_v.mean(axis=0))/df_worldnews_v.std(axis=0)
df_finance_v = (df_finance_v - df_finance_v.mean(axis=0))/df_finance_v.std(axis=0)
df_crypto_v = (df_crypto_v - df_crypto_v.mean(axis=0))/df_crypto_v.std(axis=0)
df_wsb_v = (df_wsb_v - df_wsb_v.mean(axis=0))/df_wsb_v.std(axis=0)
df_investing_v = (df_investing_v - df_investing_v.mean(axis=0))/df_investing_v.std(axis=0)

tbill['10y Tbill'] = (tbill['10y Tbill'] - tbill['10y Tbill'].mean(axis=0))/tbill['10y Tbill'].std(axis=0)
vix['VIX Index'] = (vix['VIX Index'] - vix['VIX Index'].mean(axis=0))/vix['VIX Index'].std(axis=0)

sp500_lag1 = (sp500_lag1 - sp500_lag1.mean(axis=0))/sp500_lag1.std(axis=0)
tesla_lag1 = (tesla_lag1 - tesla_lag1.mean(axis=0))/tesla_lag1.std(axis=0)
btc_lag1 = (btc_lag1 - btc_lag1.mean(axis=0))/btc_lag1.std(axis=0)

"""With Headlines and TextBlob"""
df_sp500_t = pd.concat([df_worldnews_t.shift(1, axis = 0).dropna(), df_finance_t.shift(1, axis = 0).dropna(), df_crypto_t.shift(1, axis = 0).dropna(), 
                        df_wsb_t.shift(1, axis = 0).dropna(), df_investing_t.shift(1, axis = 0).dropna(), 
                        vix['VIX Index'].shift(1, axis = 0).dropna(), tbill['10y Tbill'].shift(1, axis = 0).dropna(),
                        sp500_lag1, tesla_lag1, btc_lag1, freq_headlines.shift(1, axis = 0).dropna(), sp500['SP500 Jump']], axis=1).dropna()
df_sp500_t.name = 'df_sp500_t'

df_tesla_t = pd.concat([df_worldnews_t.shift(1, axis = 0).dropna(), df_finance_t.shift(1, axis = 0).dropna(), df_crypto_t.shift(1, axis = 0).dropna(), 
                        df_wsb_t.shift(1, axis = 0).dropna(), df_investing_t.shift(1, axis = 0).dropna(), 
                        vix['VIX Index'].shift(1, axis = 0).dropna(), tbill['10y Tbill'].shift(1, axis = 0).dropna(),
                        sp500_lag1, tesla_lag1, btc_lag1, freq_headlines.shift(1, axis = 0).dropna(), tesla['TSLA Jump']], axis=1).dropna()
df_tesla_t.name = 'df_tesla_t'

df_btc_t = pd.concat([df_worldnews_t.shift(1, axis = 0).dropna(), df_finance_t.shift(1, axis = 0).dropna(), df_crypto_t.shift(1, axis = 0).dropna(), 
                      df_wsb_t.shift(1, axis = 0).dropna(), df_investing_t.shift(1, axis = 0).dropna(), 
                      vix['VIX Index'].shift(1, axis = 0).dropna(), tbill['10y Tbill'].shift(1, axis = 0).dropna(),
                      sp500_lag1, tesla_lag1, btc_lag1, freq_headlines.shift(1, axis = 0).dropna(), btc['BTC Jump']], axis=1).dropna()
df_btc_t.name = 'df_btc_t'

"""With Headlines and Vader"""
df_sp500_v = pd.concat([df_worldnews_v.shift(1, axis = 0).dropna(), df_finance_v.shift(1, axis = 0).dropna(), df_crypto_v.shift(1, axis = 0).dropna(), 
                        df_wsb_v.shift(1, axis = 0).dropna(), df_investing_v.shift(1, axis = 0).dropna(), 
                        vix['VIX Index'].shift(1, axis = 0).dropna(), tbill['10y Tbill'].shift(1, axis = 0).dropna(),
                        sp500_lag1, tesla_lag1, btc_lag1, freq_headlines.shift(1, axis = 0).dropna(), sp500['SP500 Jump']], axis=1).dropna()
df_sp500_v.name = 'df_sp500_v'

df_tesla_v = pd.concat([df_worldnews_v.shift(1, axis = 0).dropna(), df_finance_v.shift(1, axis = 0).dropna(), df_crypto_v.shift(1, axis = 0).dropna(), 
                        df_wsb_v.shift(1, axis = 0).dropna(), df_investing_v.shift(1, axis = 0).dropna(), 
                        vix['VIX Index'].shift(1, axis = 0).dropna(), tbill['10y Tbill'].shift(1, axis = 0).dropna(),
                        sp500_lag1, tesla_lag1, btc_lag1, freq_headlines.shift(1, axis = 0).dropna(), tesla['TSLA Jump']], axis=1).dropna()
df_tesla_v.name = 'df_tesla_v'

df_btc_v = pd.concat([df_worldnews_v.shift(1, axis = 0).dropna(), df_finance_v.shift(1, axis = 0).dropna(), df_crypto_v.shift(1, axis = 0).dropna(), 
                      df_wsb_v.shift(1, axis = 0).dropna(), df_investing_v.shift(1, axis = 0).dropna(), 
                      vix['VIX Index'].shift(1, axis = 0).dropna(), tbill['10y Tbill'].shift(1, axis = 0).dropna(),
                      sp500_lag1, tesla_lag1, btc_lag1, freq_headlines.shift(1, axis = 0).dropna(), btc['BTC Jump']], axis=1).dropna()
df_btc_v.name = 'df_btc_v'

"""Without  Headlines"""
df_sp500_temp = pd.concat([vix['VIX Index'].shift(1, axis = 0).dropna(), tbill['10y Tbill'].shift(1, axis = 0).dropna(), 
                           sp500_lag1, tesla_lag1, btc_lag1, sp500['SP500 Jump'], df_btc_v['BTC Jump']], axis=1).dropna()
df_sp500 = df_sp500_temp.iloc[:, 0:df_sp500_temp.shape[1]-1].copy()
df_sp500.name = 'df_sp500'  
           
df_tesla_temp = pd.concat([vix['VIX Index'].shift(1, axis = 0).dropna(), tbill['10y Tbill'].shift(1, axis = 0).dropna(), 
                      sp500_lag1, tesla_lag1, btc_lag1, tesla['TSLA Jump'], df_btc_v['BTC Jump']], axis=1).dropna()
df_tesla = df_tesla_temp.iloc[:, 0:df_tesla_temp.shape[1]-1].copy()
df_tesla.name = 'df_tesla'  
              
df_btc_temp = pd.concat([vix['VIX Index'].shift(1, axis = 0).dropna(), tbill['10y Tbill'].shift(1, axis = 0).dropna(), 
                    sp500_lag1, tesla_lag1, btc_lag1, btc['BTC Jump'], df_btc_v['BTC Jump']], axis=1).dropna()
df_btc = df_btc_temp.iloc[:, 0:df_btc_temp.shape[1]-1].copy()
df_btc.name = 'df_btc' 

# Keep the same dataframe shape for the returns as well 
sp500_temp = pd.concat([sp500['SP500 Returns'], df_sp500['SP500 Jump']], axis=1).dropna()      
tesla_temp = pd.concat([tesla['TSLA Returns'], df_tesla['TSLA Jump']], axis=1).dropna()
btc_temp = pd.concat([btc['BTC Returns'], df_btc['BTC Jump']], axis=1).dropna()      

# =============================================================================
# PART 6: Run all predictions 
# =============================================================================

def prediction(dataframe, hpt, measure_pred):
    """
    This function will run all our selected predictions defined on prediction_models.py

    Parameters
    ----------
    dataframe : DataFrame
        This DF will include the aforementioned inputs, and the last column is our binary output.
    hpt : String
        This variable define which hyperparameter tuning to select [No hpt, grid search, random search].
    measure_pred : String
        It is the measure selected to classify our models performances from worst to best.
        [accuracy, recall, precision, f1, mse, mae]

    Returns
    -------
    - Dataframe returning the metrics of our models; 
    - Dataframe returning the prediction of our models.

    """
        
    acc_lr, pre_lr, rec_lr, f1_lr, mse_lr, mae_lr = [], [], [], [], [], []
    #acc_gboost, pre_gboost, rec_gboost, f1_gboost, mse_gboost, mae_gboost = [], [], [], [], [], []
    acc_ada, pre_ada, rec_ada, f1_ada, mse_ada, mae_ada = [], [], [], [], [], []
    acc_svm, pre_svm, rec_svm, f1_svm, mse_svm, mae_svm = [], [], [], [], [], []
    #acc_sgd, pre_sgd, rec_sgd, f1_sgd, mse_sgd, mae_sgd = [], [], [], [], [], []
    acc_knn, pre_knn, rec_knn, f1_knn, mse_knn, mae_knn = [], [], [], [], [], []
    acc_forest, pre_forest, rec_forest, f1_forest, mse_forest, mae_forest = [], [], [], [], [], []
    #acc_mlp, pre_mlp, rec_mlp, f1_mlp, mse_mlp, mae_mlp = [], [], [], [], [], []
    acc_nb, pre_nb, rec_nb, f1_nb, mse_nb, mae_nb = [], [], [], [], [], []
    acc_ann, pre_ann, rec_ann, f1_ann, mse_ann, mae_ann = [], [], [], [], [], []
    acc_lstm, pre_lstm, rec_lstm, f1_lstm, mse_lstm, mae_lstm = [], [], [], [], [], [] 
    
    X = dataframe.iloc[:, 0:(dataframe.shape[1]- 1)]
    y = dataframe.iloc[:, -1]
    
    #Split into in-sample (before January 2020) and out-sample (after January 2020)
    X_train_df = X.loc[(X.index < pd.to_datetime('2020-01-01'))].iloc[:,:]
    X_train = X_train_df.values
    y_train_df = y.loc[(y.index < pd.to_datetime('2020-01-01'))]
    y_train = y_train_df.values
    
    X_test_df = X.loc[(X.index >= pd.to_datetime('2020-01-01'))].iloc[:,:]
    X_test = X_test_df.values
    y_test_df = y.loc[(y.index >= pd.to_datetime('2020-01-01'))]
    y_test = y_test_df.values

    ###Logistic regression###
    lr = logistic(hpt, X_train, X_test, y_train, y_test, measure_pred.lower())
    
    scores = [acc_lr, pre_lr, rec_lr, f1_lr, mse_lr, mae_lr]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(lr[j])
        
    y_pred_lr = lr[-1]

    ###Gradient Boosting###
    # This model has been ignored due to a long execution time in hyperparameter tuning.
    """
    #Execution time too long during GridSearch/RandomSearch
    gboost = gradboost(hpt, X_train, X_test, y_train, y_test, measure_pred.lower())
    gboost = logistic(hpt, X_train, X_test, y_train, y_test)
    
    scores = [acc_gboost, pre_gboost, rec_gboost, f1_gboost, mse_gboost, mae_gboost]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(gboost[j])

    y_pred_gboost = gboost[-1]
    """
    ###ADA Boosting###
    ada = adaboost(hpt, X_train, X_test, y_train, y_test, measure_pred.lower())
    
    scores = [acc_ada, pre_ada, rec_ada, f1_ada, mse_ada, mae_ada]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(ada[j])

    y_pred_ada = ada[-1]
    
    ###SVM###
    svmt = svm(hpt, X_train, X_test, y_train, y_test, measure_pred.lower())
    
    scores = [acc_svm, pre_svm, rec_svm, f1_svm, mse_svm, mae_svm]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(svmt[j])
        
    y_pred_svmt = svmt[-1]
    
    ###SGDClassifier###
    # Although it gives great results, this model has been ignored for the repport since it
    # does not gives consistently the same results between each run (the seed is affected).
    """
    sgdt = sgd(hpt, X_train, X_test, y_train, y_test, measure_pred.lower())
    
    scores = [acc_sgd, pre_sgd, rec_sgd, f1_sgd, mse_sgd, mae_sgd]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(sgdt[j])
        
    y_pred_sgdt = sgdt[-1]
    """
    ###KNN###
    knnt = knn(hpt, X_train, X_test, y_train, y_test, measure_pred.lower())
    
    scores = [acc_knn, pre_knn, rec_knn, f1_knn, mse_knn, mae_knn]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(knnt[j])
    
    y_pred_knnt = knnt[-1]
    
    ###Random Forest###
    forest = randomforest(hpt, X_train, X_test, y_train, y_test, measure_pred.lower())
    
    scores = [acc_forest, pre_forest, rec_forest, f1_forest, mse_forest, mae_forest]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(forest[j])
    
    y_pred_forest = forest[-1]
    
    ###MLP###
    # Although it gives great results, this model has been ignored for the repport since it
    # does not gives consistently the same results between each run (the seed is affected).
    """
    mlpt = mlp(hpt, X_train, X_test, y_train, y_test, measure_pred.lower())
    
    scores = [acc_mlp, pre_mlp, rec_mlp, f1_mlp, mse_mlp, mae_mlp]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(mlpt[j])
        
    y_pred_mlpt = mlpt[-1]
    """
    ###Naive Bayes###
    nb = naivebayes(hpt, X_train, X_test, y_train, y_test, measure_pred.lower())
    
    scores = [acc_nb, pre_nb, rec_nb, f1_nb, mse_nb, mae_nb]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(nb[j])
    
    y_pred_nb = nb[-1]
 
    ###ANN###
    annt = ann(X_train, X_test, y_train, y_test, dataframe.name+'_'+hpt)
    
    scores = [acc_ann, pre_ann, rec_ann, f1_ann, mse_ann, mae_ann]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(annt[j])
        
    y_pred_annt = annt[-1]
    
    ###LSTM###
    lstmt = lstm(dataframe, dataframe.name+'_'+hpt)
    
    scores = [acc_lstm, pre_lstm, rec_lstm, f1_lstm, mse_lstm, mae_lstm]
    for i, j in zip(scores, range(0, len(scores))):
        i.append(lstmt[j])
        
    y_pred_lstmt = lstmt[-1]
                        
    ###Collect All Results###
    d1 = {"LR": [np.mean(acc_lr), np.mean(pre_lr), np.mean(rec_lr), np.mean(f1_lr), np.mean(mse_lr), np.mean(mae_lr)],
         #"Gradient Boosting": [np.mean(acc_gboost), np.mean(pre_gboost), np.mean(rec_gboost), np.mean(f1_gboost), np.mean(mse_gboost), np.mean(mae_gboost)],
         "ADA Boost.": [np.mean(acc_ada), np.mean(pre_ada), np.mean(rec_ada), np.mean(f1_ada), np.mean(mse_ada), np.mean(mae_ada)],
         "SVM": [np.mean(acc_svm), np.mean(pre_svm), np.mean(rec_svm), np.mean(f1_svm), np.mean(mse_svm), np.mean(mae_svm)],
         #"SGD": [np.mean(acc_sgd), np.mean(pre_sgd), np.mean(rec_sgd), np.mean(f1_sgd), np.mean(mse_sgd), np.mean(mae_sgd)],
         "KNN": [np.mean(acc_knn), np.mean(pre_knn), np.mean(rec_knn), np.mean(f1_knn), np.mean(mse_knn), np.mean(mae_knn)],
         "RF": [np.mean(acc_forest), np.mean(pre_forest), np.mean(rec_forest), np.mean(f1_forest), np.mean(mse_forest), np.mean(mae_forest)],
         #"MLP": [np.mean(acc_mlp), np.mean(pre_mlp), np.mean(rec_mlp), np.mean(f1_mlp), np.mean(mse_mlp), np.mean(mae_mlp)],
         "NB": [np.mean(acc_nb), np.mean(pre_nb), np.mean(rec_nb), np.mean(f1_nb), np.mean(mse_nb), np.mean(mae_nb)],
         "ANN": [np.mean(acc_ann), np.mean(pre_ann), np.mean(rec_ann), np.mean(f1_ann), np.mean(mse_ann), np.mean(mae_ann)],
         "LSTM": [np.mean(acc_lstm), np.mean(pre_lstm), np.mean(rec_lstm), np.mean(f1_lstm), np.mean(mse_lstm), np.mean(mae_lstm)],
         }
    
    d2 = {"Actual": y_test,
         "LR": y_pred_lr,
         #"Gradient Boosting": y_pred_gboost,
         "ADA Boost.": y_pred_ada,
         "SVM": y_pred_svmt,
         #"SGD": y_pred_sgdt,
         "KNN": y_pred_knnt,
         "RF": y_pred_forest,
         #"MLP": y_pred_mlpt,
         "NB": y_pred_nb,
         "ANN": np.concatenate(y_pred_annt, axis=0),
         "LSTM": np.concatenate(y_pred_lstmt, axis=0),
         }
    
    index = ['Accuracy', 'Precision', 'Recall', 'F1', 'MSE', 'MAE']
    
    df_measures = pd.DataFrame(data = d1, index=index)
    df_measures['Max Value'] = df_measures.idxmax(axis=1)
    df_y = pd.DataFrame(data = d2, index=y_test_df.index)
         
    return (df_measures.dropna(axis=1, how='any'), df_y.dropna())

# ==========================================================================================
# Part 7: Evalutation of our Predictions (Long/Short Positions) Compared to Long-Hold Only         
# ==========================================================================================

def hit_ratio(return_dataset):
    """
    This function determine the hit ratio of any time series returns

    Parameters
    ----------
    return_dataset : Dataframe
        The returns of the asset.

    Returns
    -------
    TYPE
        It returns the hit ratio.

    """
    return len(return_dataset[return_dataset >= 0]) / len(return_dataset)

def max_drawdown(cum_returns):
    """
    It determines the maximum drawdown over the cumulative returns
    of a time series.

    Parameters
    ----------
    cum_returns : Dataframe
        Cumulative Return.

    Returns
    -------
    max_monthly_drawdown : TYPE
        Evolution of the max drawdown (negative output).

    """
    roll_max = cum_returns.cummax()
    monthly_drawdown = cum_returns/roll_max - 1
    max_monthly_drawdown = monthly_drawdown.cummin()
    return max_monthly_drawdown

def perf(data, name):
    """
    This function compute all the required performances of a daily time series.

    Parameters
    ----------
    data : Dataframe
        Returns of a given portfolio.
    benchmark : Dataframe
        Returns of the benchmark.
    name : String
        Name of the dataframe.

    Returns
    -------
    df : TYPE
        Return a dataframe that contains the annualized returns, volatility,
        Sharpe ratio, max drawdown and hit ratio.

    """

    exp = np.mean(data,0)*252
    vol = np.std(data,0)*np.power(252,0.5)
    sharpe = exp/vol
    max_dd = max_drawdown((data+1).cumprod())
    hit = hit_ratio(data)
    df = pd.DataFrame({name: [exp, vol, sharpe, max_dd.min(), hit]}, index = ['Returns', 'Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Hit Ratio'])
    return df

def performances(hpt, measure):
    """
    This function will create a long/short portfolio based on our predictions, and compare it to 
    a hold only portfolio.

    Parameters
    ----------
    hpt : String
        This variable define which hyperparameter tuning to select [No hpt, grid search, random search].
    measure_pred : String
        It is the measure selected to classify our models performances from worst to best
        [accuracy, recall, precision, f1, mse, mae].

    Returns
    -------
    perf_sp500_t : Dataframe
        Performances of SP500 index including reddit sentiments (Textblob).
    perf_tesla_t : Dataframe
        Performances of TSLA including reddit sentiments (Textblob).
    perf_btc_t : Dataframe
        Performances of BTCUSD including reddit sentiments (Textblob).
    perf_sp500_v : Dataframe
        Performances of SP500 index including reddit sentiments (VADER).
    perf_tesla_v : Dataframe
        Performances of TSLA including reddit sentiments (VADER).
    perf_btc_v : Dataframe
        Performances of BTCUSD including reddit sentiments (VADER).
    perf_sp500 : Dataframe
        Performances of SP500 index excluding reddit sentiments.
    perf_tesla : Dataframe
        Performances of TSLA excluding reddit sentiments.
    perf_btc : Dataframe
        Performances of BTCUSD excluding reddit sentiments.
    """
    
    #Create files in the working directory
    if not os.path.isdir(f'Plot_{measure}'):
        os.makedirs(f'Plot_{measure}')
               
    #Create files in the working directory
    if not os.path.isdir(f'Output_{measure}'):
        os.makedirs(f'Output_{measure}')
    
    """Run all predictions with headlines (Textblob)"""
    df_sp500_t_measures, df_sp500_t_pred = prediction(df_sp500_t, hpt, measure)
    df_sp500_t_pred.replace(0, -1, inplace=True)  
    
    df_tesla_t_measures, df_tesla_t_pred = prediction(df_tesla_t, hpt, measure)
    df_tesla_t_pred.replace(0, -1, inplace=True)
    
    df_btc_t_measures, df_btc_t_pred = prediction(df_btc_t, hpt, measure) 
    df_btc_t_pred.replace(0, -1, inplace=True) 
    
    """Run all predictions with headlines (Vader)"""
    df_sp500_v_measures, df_sp500_v_pred = prediction(df_sp500_v, hpt, measure)  
    df_sp500_v_pred.replace(0, -1, inplace=True)  
    
    df_tesla_v_measures, df_tesla_v_pred = prediction(df_tesla_v, hpt, measure)
    df_tesla_v_pred.replace(0, -1, inplace=True)  
    
    df_btc_v_measures, df_btc_v_pred = prediction(df_btc_v, hpt, measure)  
    df_btc_v_pred.replace(0, -1, inplace=True)  
                
    """Run all predictions without headlines"""
    df_sp500_measures, df_sp500_pred = prediction(df_sp500, hpt, measure)  
    df_sp500_pred.replace(0, -1, inplace=True)  
    
    df_tesla_measures, df_tesla_pred = prediction(df_tesla, hpt, measure)
    df_tesla_pred.replace(0, -1, inplace=True)
    
    df_btc_measures, df_btc_pred = prediction(df_btc, hpt, measure)  
    df_btc_pred.replace(0, -1, inplace=True)
    
    ###Evalutation of the Prediciton Compared to Long-Hold Only###
    
    """Long Only Dataframe"""
    sp500_OS = sp500_temp.loc[sp500_temp.index >= pd.to_datetime('2020-01-01'), 'SP500 Returns'].dropna()
    tesla_OS = tesla_temp.loc[tesla_temp.index >= pd.to_datetime('2020-01-01'), 'TSLA Returns'].dropna()
    btc_OS = btc_temp.loc[btc_temp.index >= pd.to_datetime('2020-01-01'), 'BTC Returns'].dropna()
    
    """TextBlob"""
    sp500_OS_pred_t = (sp500_OS*df_sp500_t_pred[df_sp500_t_measures['Max Value'][measure]]).dropna()
    tesla_OS_pred_t = (tesla_OS*df_tesla_t_pred[df_tesla_t_measures['Max Value'][measure]]).dropna()
    btc_OS_pred_t = (btc_OS*df_btc_t_pred[df_btc_t_measures['Max Value'][measure]]).dropna()  
    
    plt.figure(figsize=(15,7))
    plt.plot(cum_prod(sp500_OS), 'b', label='Hold Only (Reality)')
    plt.plot(cum_prod(sp500_OS_pred_t), 'r', label='Long/Short (Predictions)')
    plt.title('S&P500 with Reddit Posts (TextBlob Sentiment) using ' + df_sp500_t_measures['Max Value'][measure] + ' (' + measure + ': ' + str(df_sp500_t_measures[df_sp500_t_measures['Max Value'][measure]][measure])+')')
    plt.legend(loc='upper left')
    plt.savefig(f'Plot_{measure}/sp500_cumul_perf_textblob_{hpt}.png')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,7))
    plt.plot(cum_prod(tesla_OS), 'b', label='Hold Only (Reality)')
    plt.plot(cum_prod(tesla_OS_pred_t), 'r', label='Long/Short (Predictions)')
    plt.title('TSLA with Reddit Posts (TextBlob Sentiment) using ' + df_tesla_t_measures['Max Value'][measure] + ' (' + measure + ': ' + str(df_tesla_t_measures[df_tesla_t_measures['Max Value'][measure]][measure])+')')
    plt.legend(loc='upper left')
    plt.savefig(f'Plot_{measure}/tsla_cumul_perf_textblob_{hpt}.png')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,7))
    plt.plot(cum_prod(btc_OS), 'b', label='Hold Only (Reality)')
    plt.plot(cum_prod(btc_OS_pred_t), 'r', label='Long/Short (Predictions)')
    plt.title('BTC-USD with Reddit Posts (TextBlob Sentiment) using ' + df_btc_t_measures['Max Value'][measure] + ' (' + measure + ': ' + str(df_btc_t_measures[df_btc_t_measures['Max Value'][measure]][measure])+')')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'Plot_{measure}/btc_cumul_perf_textblob_{hpt}.png')
    plt.show()
    plt.close()
        
    """Vader"""
    sp500_OS_pred_v = (sp500_OS*df_sp500_v_pred[df_sp500_v_measures['Max Value'][measure]]).dropna() 
    tesla_OS_pred_v = (tesla_OS*df_tesla_v_pred[df_tesla_v_measures['Max Value'][measure]]).dropna()
    btc_OS_pred_v = (btc_OS*df_btc_v_pred[df_btc_v_measures['Max Value'][measure]]).dropna()  
    
    plt.figure(figsize=(15,7))
    plt.plot(cum_prod(sp500_OS), 'b', label='Hold Only (Reality)')
    plt.plot(cum_prod(sp500_OS_pred_v), 'r', label='Long/Short (Predictions)')
    plt.title('S&P500 with Reddit Posts (Vader Sentiment) using ' + df_sp500_v_measures['Max Value'][measure] + ' (' + measure + ': ' + str(df_sp500_v_measures[df_sp500_v_measures['Max Value'][measure]][measure])+')')
    plt.legend(loc='upper left')
    plt.savefig(f'Plot_{measure}/sp500_cumul_perf_vader_{hpt}.png')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,7))
    plt.plot(cum_prod(tesla_OS), 'b', label='Hold Only (Reality)')
    plt.plot(cum_prod(tesla_OS_pred_v), 'r', label='Long/Short (Predictions)')
    plt.title('TSLA with Reddit Posts (Vader Sentiment) using ' + df_tesla_v_measures['Max Value'][measure] + ' (' + measure + ': ' + str(df_tesla_v_measures[df_tesla_v_measures['Max Value'][measure]][measure])+')')
    plt.legend(loc='upper left')
    plt.savefig(f'Plot_{measure}/tsla_cumul_perf_vader_{hpt}.png')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,7))
    plt.plot(cum_prod(btc_OS), 'b', label='Hold Only (Reality)')
    plt.plot(cum_prod(btc_OS_pred_v), 'r', label='Long/Short (Predictions)')
    plt.title('BTC-USD with Reddit Posts (Vader Sentiment) using ' + df_btc_v_measures['Max Value'][measure] + ' (' + measure + ': ' + str(df_btc_v_measures[df_btc_v_measures['Max Value'][measure]][measure])+')')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'Plot_{measure}/btc_cumul_perf_vader_{hpt}.png')
    plt.show()
    plt.close()
    
    """No Sentiments"""
    sp500_OS_pred = (sp500_OS*df_sp500_pred[df_sp500_measures['Max Value'][measure]]).dropna() 
    tesla_OS_pred = (tesla_OS*df_tesla_pred[df_tesla_measures['Max Value'][measure]]).dropna()
    btc_OS_pred = (btc_OS*df_btc_pred[df_btc_measures['Max Value'][measure]]).dropna()  
    
    plt.figure(figsize=(15,7))
    plt.plot(cum_prod(sp500_OS), 'b', label='Hold Only (Reality)')
    plt.plot(cum_prod(sp500_OS_pred), 'r', label='Long/Short (Predictions)')
    plt.title('S&P500 without Reddit Posts using ' + df_sp500_measures['Max Value'][measure] + ' (' + measure + ': ' + str(df_sp500_measures[df_sp500_measures['Max Value'][measure]][measure])+')')
    plt.legend(loc='upper left')
    plt.savefig(f'Plot_{measure}/sp500_cumul_perf_{hpt}.png')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,7))
    plt.plot(cum_prod(tesla_OS), 'b', label='Hold Only (Reality)')
    plt.plot(cum_prod(tesla_OS_pred), 'r', label='Long/Short (Predictions)')
    plt.title('TSLA without Reddit Posts using ' + df_tesla_measures['Max Value'][measure] + ' (' + measure + ': ' + str(df_tesla_measures[df_tesla_measures['Max Value'][measure]][measure])+')')
    plt.legend(loc='upper left')
    plt.savefig(f'Plot_{measure}/tsla_cumul_perf_{hpt}.png')
    plt.show()
    plt.close()
    
    plt.figure(figsize=(15,7))
    plt.plot(cum_prod(btc_OS), 'b', label='Hold Only (Reality)')
    plt.plot(cum_prod(btc_OS_pred), 'r', label='Long/Short (Predictions)')
    plt.title('BTC-USD without Reddit Posts using ' + df_btc_measures['Max Value'][measure] + ' (' + measure + ': ' + str(df_btc_measures[df_btc_measures['Max Value'][measure]][measure])+')')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'Plot_{measure}/btc_cumul_perf_{hpt}.png')
    plt.show()
    plt.close()
        
    ###Get some statistics regarding the portfolios###
    """TextBlob"""
    perf_sp500_t = pd.concat([perf(sp500_OS, 'SP500 Hold'), perf(sp500_OS_pred_t, 'Long/Short (Texblob)')], axis=1) 
    perf_sp500_t.to_latex(f'Output_{measure}/perf_sp500_t_{hpt}.tex')

    perf_tesla_t = pd.concat([perf(tesla_OS, 'TSLA Hold'), perf(tesla_OS_pred_t, 'Long/Short (Texblob)')], axis=1)
    perf_tesla_t.to_latex(f'Output_{measure}/perf_tesla_t_{hpt}.tex')
    
    perf_btc_t = pd.concat([perf(btc_OS, 'BTCUSD Hold'), perf(btc_OS_pred_t, 'Long/Short (Texblob)')], axis=1)
    perf_btc_t.to_latex(f'Output_{measure}/perf_btc_t_{hpt}.tex')
    
    """Vader"""
    perf_sp500_v = pd.concat([perf(sp500_OS, 'SP500 Hold'), perf(sp500_OS_pred_v, 'Long/Short (Vader)')], axis=1)
    perf_sp500_v.to_latex(f'Output_{measure}/perf_sp500_v_{hpt}.tex')

    perf_tesla_v = pd.concat([perf(tesla_OS, 'TSLA Hold'), perf(tesla_OS_pred_v, 'Long/Short (Vader)')], axis=1)
    perf_tesla_v.to_latex(f'Output_{measure}/perf_tesla_v_{hpt}.tex')
    
    perf_btc_v = pd.concat([perf(btc_OS, 'BTCUSD Hold'), perf(btc_OS_pred_v, 'Long/Short (Vader)')], axis=1)
    perf_btc_v.to_latex(f'Output_{measure}/perf_btc_v_{hpt}.tex')
    
    """Vader"""
    perf_sp500 = pd.concat([perf(sp500_OS, 'SP500 Hold'), perf(sp500_OS_pred, 'Long/Short (No Sentiment)')], axis=1)
    perf_sp500.to_latex(f'Output_{measure}/perf_sp500_{hpt}.tex')

    perf_tesla = pd.concat([perf(tesla_OS, 'TSLA Hold'), perf(tesla_OS_pred, 'Long/Short (No Sentiment)')], axis=1)
    perf_tesla.to_latex(f'Output_{measure}/perf_tesla_{hpt}.tex')
    
    perf_btc = pd.concat([perf(btc_OS, 'BTCUSD Hold'), perf(btc_OS_pred, 'Long/Short (No Sentiment)')], axis=1)
    perf_btc.to_latex(f'Output_{measure}/perf_btc_{hpt}.tex')
    
    return (perf_sp500_t, perf_tesla_t, perf_btc_t, perf_sp500_v, perf_tesla_v, perf_btc_v, perf_sp500, perf_tesla, perf_btc)

# =============================================================================
# PART 8: RUN ALL MODELS 
# =============================================================================

#Create files in the working directory
if not os.path.isdir('Plot_model'):
    os.makedirs('Plot_model')

"""No Hyperparameter Tuning"""
perf_sp500_base_t_acc, perf_tesla_base_t_acc, perf_btc_base_t_acc, perf_sp500_base_v_acc, perf_tesla_base_v_acc, perf_btc_base_v_acc, perf_sp500_base_acc, perf_tesla_base_acc, perf_btc_base_acc  =  performances('baseline', 'Accuracy')

"""GridSearch Hyperparameter Tuning"""
perf_sp500_grid_t_acc, perf_tesla_grid_t_acc, perf_btc_grid_t_acc, perf_sp500_grid_v_acc, perf_tesla_grid_v_acc, perf_btc_grid_v_acc, perf_sp500_grid_acc, perf_tesla_grid_acc, perf_btc_grid_acc = performances('grid_search', 'Accuracy')

"""RandomSearch Hyperparameter Tuning"""
perf_sp500_random_t_acc, perf_tesla_random_t_acc, perf_btc_random_t_acc, perf_sp500_random_v_acc, perf_tesla_random_v_acc, perf_btc_random_v_acc, perf_sp500_random_acc, perf_tesla_random_acc, perf_btc_random_acc = performances('random_search', 'Accuracy')

os.rename(r'Plot_model',r'Plot_model_Accuracy')



