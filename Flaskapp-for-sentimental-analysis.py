import os
import requests
import operator
import re
import nltk
from flask import Flask, render_template, request
import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

nltk.download("vader_lexicon")
import pandas as pd
import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer as vader_sentiment
from pandas_datareader import data as pdr
import yfinance as yfinance_reader
from sklearn import preprocessing




def update_sentiment_value(input_df_sentiment):
    input_df_sentiment["Score"] = 0.0
    for i, row in input_df_sentiment.iterrows():
        curr_score = float(
            vader_sentiment().polarity_scores(row["Headlines"])["compound"]
        )
        input_df_sentiment.at[i, "Score"] = curr_score
        input_df_sentiment.at[i, "Dates"] = datetime.datetime.strptime(
            str(row["Dates"]), "%Y-%m-%d"
        ).date()
    return input_df_sentiment


def load_TSLA_by_yfinance_Data(ticker_name):
    yfinance_reader.pdr_override()
    TSLA_Yahoo_data = pdr.get_data_yahoo(
        ticker_name, start="2018-02-13", end="2020-04-27"
    )
    TSLA_Yahoo_data["Returns"] = TSLA_Yahoo_data["Adj Close"] / (TSLA_Yahoo_data["Adj Close"].shift(1) - 1)
    return TSLA_Yahoo_data


def get_mean_score_by_date(input_df_sentiment):
    df_sentiment_mean_by_day = input_df_sentiment.groupby(["Dates"]).mean()
    df_sentiment_mean_by_day['Score'] = df_sentiment_mean_by_day['Score'].shift(1)
    return df_sentiment_mean_by_day


def compare_sentiment_returns(df_sentiment, df_yahoo):
    merged_df = pd.merge(
        df_yahoo[["Returns"]],
        df_sentiment[["Score"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    return merged_df


def preprocessings(df):
    x = df[['Score']].values
    y = df[['Returns']].values

    x = preprocessing.scale(x)
    y = preprocessing.scale(y)
    values = []
    for val in y:
        if str(val).startswith("[-"):
            values.append(0)
        else:
            values.append(1)

    df['Ret'] = values
    df.drop('Returns', axis=1, inplace=True)
    x = x = df[['Score']].values
    y = df[['Ret']].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test


def knc(x_train, x_test, y_train, y_test):
    model1 = KNeighborsClassifier(algorithm='ball_tree')
    model1.fit(x_train, y_train)
    predictions = model1.predict(x_test)
    knc_score = accuracy_score(y_test, predictions) * 100
    return(knc_score)



def logisticreg(x_train, x_test, y_train, y_test):
    model2 = LogisticRegression(solver='newton-cg', multi_class='ovr', max_iter=200, penalty='l2')
    model2.fit(x_train, y_train)
    predictions = model2.predict(x_test)
    LogRegression = accuracy_score(y_test, predictions) * 100
    return(LogRegression)

def support(x_train, x_test, y_train, y_test):
    model3 = svm.SVC(kernel='sigmoid')
    model3.fit(x_train, y_train)
    predictions = model3.predict(x_test)
    supp = accuracy_score(y_test, predictions) * 100
    return(supp)


def naive_bayes_gaussian(x_train, x_test, y_train, y_test):
    model_guass = GaussianNB()
    model_guass.fit(x_train, y_train)
    predictions_gauss = model_guass.predict(x_test)
    nbg = accuracy_score(y_test, predictions_gauss) * 100
    return nbg


def naive_bayes_bernoulli(x_train, x_test, y_train, y_test):
    model4 = BernoulliNB()
    model4.fit(x_train, y_train)
    predictions = model4.predict(x_test)
    nbb = accuracy_score(y_test, predictions) * 100
    return nbb

app = Flask(__name__)

@app.route('/')
def home():
    print('***************************************')
    input_datafile_csv = "TSLA_1.csv"
    df_grabbed_sentiment = pd.read_csv(input_datafile_csv)
    df_grabbed_sentiment = update_sentiment_value(df_grabbed_sentiment)
    TSLA_Yahoo_data = load_TSLA_by_yfinance_Data("TSLA")
    df_mean_score = get_mean_score_by_date(df_grabbed_sentiment)
    TSLA_Yahoo_data.to_csv("TSLA_Yahoo_data")
    merged_df = compare_sentiment_returns(df_mean_score, TSLA_Yahoo_data)
    merged_df = merged_df.fillna(merged_df.mean())
    a, b, c, d = preprocessings(merged_df)
    kneigh = knc(a, b, c, d)
    logistic = logisticreg(a, b, c, d)
    supportVector = support(a, b, c, d)
    naivebgauss = naive_bayes_gaussian(a,b,c,d)
    naivebberno = naive_bayes_bernoulli(a,b,c,d)

    return render_template('home.html' , data1= kneigh , data2= logistic , data3= supportVector , data4= naivebgauss , data5= naivebberno)

if __name__ == '__main__':
    app.run(debug=True)