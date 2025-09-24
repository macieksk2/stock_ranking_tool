# -*- coding: utf-8 -*-
"""
Created on Sun Sep 21 20:53:00 2025

@author: macie
"""
import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
}

inv_websites = {    'MSFT' : 'https://www.microsoft.com/en-us/investor/events/fy-2025/earnings-fy-2025-q4',
                    'AAPL' : 'https://www.fool.com/earnings/call-transcripts/2025/08/01/apple-aapl-q3-2025-earnings-call-transcript/',
                    'TSLA' : 'https://www.fool.com/earnings/call-transcripts/2025/07/23/tesla-tsla-q2-2025-earnings-call-transcript/',
                    'GOOGL': 'https://www.fool.com/earnings/call-transcripts/2025/07/23/alphabet-googl-q2-2025-earnings-call-transcript/',
                    'AMZN' : 'https://www.fool.com/earnings/call-transcripts/2025/02/06/amazoncom-amzn-q4-2024-earnings-call-transcript/',
                    'NVDA' : 'https://www.fool.com/earnings/call-transcripts/2025/02/26/nvidia-nvda-q4-2025-earnings-call-transcript/',
                    'META' : 'https://www.fool.com/earnings/call-transcripts/2025/01/29/meta-platforms-meta-q4-2024-earnings-call-transcri/',
                    'ASML' : 'https://www.asml.com/en/technology/how-we-innovate',
                    'AMD'  : 'https://www.fool.com/earnings/call-transcripts/2025/02/05/advanced-micro-devices-amd-q4-2024-earnings-call-t/',
                    'ARM'  : 'https://www.fool.com/earnings/call-transcripts/2025/02/05/arm-holdings-arm-q3-2025-earnings-call-transcript/',
                    'JPM'  : 'https://www.fool.com/earnings/call-transcripts/2025/08/04/jpmorgan-jpm-q2-2025-earnings-call-transcript/',
                    'GS'   : 'https://www.fool.com/earnings/call-transcripts/2025/07/16/goldman-sachs-gs-q2-2025-earnings-call-transcript/',
                    'INTU' : 'https://www.fool.com/earnings/call-transcripts/2025/08/21/intuit-intu-q4-2025-earnings-call-transcript/',
                    'VRTX' : 'https://www.fool.com/earnings/call-transcripts/2025/02/10/vertex-pharmaceuticals-vrtx-q4-2024-earnings-call/',
                    'C'    : 'https://www.fool.com/earnings/call-transcripts/2025/07/15/citigroup-c-q2-2025-earnings-call-transcript/',
                    'KMI'  : 'https://www.fool.com/earnings/call-transcripts/2025/07/17/kinder-morgan-kmi-q2-2025-earnings-call-transcript/',
                    'LMT'  : 'https://www.fool.com/earnings/call-transcripts/2025/07/22/lockheed-martin-lmt-q2-2025-earnings-transcript/',
                    'BATS.L'    : 'https://www.fool.com/earnings/call-transcripts/2025/02/13/british-american-tobacco-plc-bti-q4-2024-earnings/',
                    'INTC'      : 'https://www.fool.com/earnings/call-transcripts/2025/08/05/intel-intc-q2-2025-earnings-call-transcript/',
                    'BABA'      : 'https://www.fool.com/earnings/call-transcripts/2025/02/20/alibaba-group-baba-q4-2024-earnings-call-transcrip/',
                    'RIO'       : 'https://www.fool.com/earnings/call-transcripts/2025/02/19/rio-tinto-group-rio-q4-2024-earnings-call-transcri/',
                    'EQR'       : 'https://www.stockinsights.ai/us/EQR/earnings-transcript/fy25-q2-0f69',
                    'TSMC34.SA' : 'https://www.fool.com/earnings/call-transcripts/2025/01/16/taiwan-semiconductor-manufacturing-tsm-q4-2024-ear/',
                    'RACE'      : 'https://www.insidermonkey.com/blog/ferrari-n-v-nyserace-q4-2024-earnings-call-transcript-1443440/',
                    'PLTR'      : 'https://www.fool.com/earnings/call-transcripts/2025/02/04/palantir-technologies-pltr-q4-2024-earnings-call-t/',
                    'LLY'       : 'https://www.fool.com/earnings/call-transcripts/2025/02/06/eli-lilly-lly-q4-2024-earnings-call-transcript/'}#,
                    # 'HIMS' : 'https://www.msn.com/en-us/money/other/hims-hers-health-inc-nyse-hims-q1-2025-earnings-call-transcript/ar-AA1Ekhc7?ocid=finance-verthp-feeds'}
######
# CHECK THE SCORE PER LINK, SPLIT BY POS/NEG/NEUT
urls = inv_websites
for url in urls.keys():
    print(url)
    response = requests.get(urls[url], timeout=20, headers = headers)
    soup     = BeautifulSoup(response.content, 'html.parser')
    # Extract text from common tags, excluding script and style tags
    text     = ' '.join(t.get_text() for t in soup.find_all(['p', 'h1', 'h2', 'h3']))
    sia     = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)
    print('score', sentiment_score['pos'] - sentiment_score['neg'])


######
# CHECK HOW THE SAME SCORE WOULD LOOK LIKE AFTER TEXT PREPROCESSING
# https://www.datacamp.com/tutorial/text-analytics-beginners-nltk?dc_referrer=https%3A%2F%2Fwww.google.com%2F
# import libraries
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# download nltk corpus (first time only)
import nltk

# nltk.download('all')

# create preprocess_text function
def preprocess_text(text):

    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)
    
    return processed_text

# apply the function df
df = preprocess_text(text)
df

# initialize NLTK sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# create get_sentiment function
def get_sentiment(text):

    scores = analyzer.polarity_scores(text)
    return scores

# apply get_sentiment function
scores = get_sentiment(df)
scores
