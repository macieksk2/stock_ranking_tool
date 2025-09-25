# -*- coding: utf-8 -*-
"""
functions for scorecard
"""
import os
import requests 
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import yfinance as yf
import matplotlib.pyplot as plt

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
################################################# QUANT ###########################################################
def get_financials_yf(ticker, statement_type):
    """
    Fetches historical financial statements from Yahoo Finance using yfinance.
    statement_type can be 'income', 'balance', or 'cashflow' or 'marketCap'.
    """
    # Create a Ticker object for the company
    try:
        stock = yf.Ticker(ticker)
    except Exception as e:
        print(f"Error creating Ticker object for {ticker}: {e}")
        return None

    # Get the financial statement based on the specified type
    if statement_type == 'income':
        data = stock.income_stmt
    elif statement_type == 'balance':
        data = stock.balance_sheet
    elif statement_type == 'cashflow':
        data = stock.cashflow
    elif statement_type == 'info':
        data = stock.info
    else:
        print(f"Invalid statement_type: {statement_type}")
        return None
    
    if statement_type != "info":
        if data.empty:
            print(f"No data found for {ticker} for {statement_type}.")
            return None

    # The yfinance library already returns a clean DataFrame, so less manipulation is needed
    # You may want to transpose the DataFrame to match your original format
    # The default yfinance format has dates as columns, which is often preferred
    return data

def create_df_yf(tickers):
    """
    Pull all the Yahoo Finance data for all tickers into a dictionary of DataFrames.
    """
    income_dfs = {}
    balance_dfs = {}
    cash_flow_dfs = {}
    info = {}

    for ticker in tickers:
        try:
            # Pull historical data using yfinance
            income_df = get_financials_yf(ticker, 'income')
            balance_df = get_financials_yf(ticker, 'balance')
            cash_flow_df = get_financials_yf(ticker, 'cashflow')
            info_dict = get_financials_yf(ticker, 'info')

            if income_df is None or balance_df is None or cash_flow_df is None:
                continue

            # yfinance returns data with dates as columns, which is a good format.
            # You can transpose it here if you need dates as rows to match your original script.
            income_dfs[ticker] = income_df.transpose()
            balance_dfs[ticker] = balance_df.transpose()
            cash_flow_dfs[ticker] = cash_flow_df.transpose()
            info[ticker] = info_dict

        except Exception as e:
            print(f"Could not process data for {ticker}. Error: {e}")

    return income_dfs, balance_dfs, cash_flow_dfs, info
# for FMP only
API_KEY = "O7jpbikr5wCjo34KFLWi9nmU3xJ9laxh"

def get_financials_fmp(ticker, statement_type):
    """
    CURRENTLY NOT USED, YF INSTEAD
    Fetches historical financial statements from Financial Modeling Prep API.
    statement_type can be 'income-statement', 'balance-sheet-statement', or 'cash-flow-statement'.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    }
    url = f"https://financialmodelingprep.com/stable/{statement_type}?symbol={ticker}&apikey={API_KEY}"
    response = requests.get(url, timeout = 20, headers = headers)

    if response.status_code != 200:
        print(f"Error fetching data for {ticker}. Status code: {response.status_code}")
        return None

    data = response.json()
    if not data:
        print(f"No data found for {ticker} for {statement_type}.")
        return None

    df       = pd.DataFrame(data).set_index('date').transpose()
    df       = df.apply(pd.to_numeric, errors='coerce').transpose()
    df.index = pd.to_datetime(df.index)

    return df

def create_df_fmp(tickers):
    """
    CURRENTLY NOT USED, YF INSTEAD
    Pull all the FMP data for all tickers into a hash map of dfs
    """
    income_dfs    = dict()
    balance_dfs   = dict()
    cash_flow_dfs = dict()
    for ticker in tickers:
        try:
            # Pull historical data from FMP API
            income_df    = get_financials_fmp(ticker, 'income-statement')
            balance_df   = get_financials_fmp(ticker, 'balance-sheet-statement')
            cash_flow_df = get_financials_fmp(ticker, 'cash-flow-statement')
    
            if income_df is None or balance_df is None or cash_flow_df is None:
                continue
    
            # Ensure data is sorted by date for accurate growth calculations
            income_df    = income_df.sort_index()
            balance_df   = balance_df.sort_index()
            cash_flow_df = cash_flow_df.sort_index()
            
            # Add to the final list
            income_dfs[ticker]    = income_df
            balance_dfs[ticker]   = balance_df
            cash_flow_dfs[ticker] = cash_flow_df
        except Exception as e:
            print(f"Could not process data for {ticker}. Error: {e}")
        
    return income_dfs, balance_dfs, cash_flow_dfs

def data_na_neg_remove(df, metrics_pos, metrics_neg, is_info = False):
    """
    Helper function to handle NAs and negative values
    """
    # Iterate thorugh dataset and preprocess
    if not is_info:
        for metric in df.columns:
            if metric in df.columns and (metric in metrics_pos or metric in metrics_neg):
                df = df.dropna(subset=[metric])
            if metric in df.columns and metric in metrics_pos:
                df.loc[df[metric] < 0, metric] = 0
    else:
        # info is not a DF, thus different approach
        for metric in metrics_pos:
            if metric in df.keys():
                # 100 - penalize valuation metric in case of na or negative
                df[metric] = df[metric] if df[metric] > 0 and df[metric] != np.nan else 100
        for metric in metrics_neg:
            if metric in df.keys():
                df[metric] = df[metric] if df[metric] != np.nan else 100
    
    return df

def data_preprocess(income_df, balance_df, cash_flow_df, info):
    """
    Currently based on YF dataset
    """
    # Find missing values:
    # Revenue - remove if NA or neg
    income_attr_pos = ['Total Revenue']
    # EPS, FCF, Gross Profit, Net Income, EBIT - remove if NA
    income_attr_pos_neg = ['Basic EPS', 'Free Cash Flow', 'Gross Profit', 'Net Income', 'EBIT']
    # Total Assets, Total Debt, Curr Assets, Curr Liab, Inventory, Cash And Cash Equivalents, Total Liabilities Net Minority Interest - remove if NA or neg
    balance_attr_pos = ['Total Assets', 'Total Debt', 'Current Assets', 'Current Liabilities', 'Inventory',
                        'Cash And Cash Equivalents', 'Total Liabilities Net Minority Interest']
    # Stock Equity, Working Capital, Retained Earnings - remove if NA
    balance_attr_pos_neg = ['Stockholders Equity', 'Working Capital', 'Retained Earnings']
    # info
    # marketCap, priceToBook, priceToSalesTrailing12Months, beta - remove if NA or neg
    info_pos = ['marketCap', 'priceToBook', 'priceToSalesTrailing12Months', 'beta']
    # forwardPE, trailingPE, enterpriseToEbitda, enterpriseToRevenue - remove if NA
    info_pos_neg = ['forwardPE', 'trailingPE', 'enterpriseToEbitda', 'enterpriseToRevenue']
    
    # Iterate thorugh dataset and preporcess (income, balance, cash flow, info)
    post_income_df = data_na_neg_remove(income_df, income_attr_pos, income_attr_pos_neg)
    post_balance_df = data_na_neg_remove(balance_df, balance_attr_pos, balance_attr_pos_neg)
    post_cash_flow_df = data_na_neg_remove(cash_flow_df, income_attr_pos, income_attr_pos_neg)
    post_info = data_na_neg_remove(info, info_pos, info_pos_neg, True)
    
    return post_income_df, post_balance_df, post_cash_flow_df, post_info


def calc_pos_neg_metrics(df, metric, years = 3):
    """
    As of now based on three years due to YF constraint, to be changed in case other source is used
    """
    growth = 0
    if len(df) >= years:
        if df[metric].iloc[0] / df[metric].iloc[years] > 0:
            growth = ((df[metric].iloc[0] / df[metric].iloc[years])**(1/years) - 1)
        else:
            if df[metric].iloc[0] > df[metric].iloc[years]:
                # 10% in case turns positive from negative
                growth = 10
            else:
                # -10% in case turns negative from positive
                growth = -10         
    return growth
def calc_income_fcf_metrics(income_df, fcf_df, years = 3):
    """
    As of now based on three years due to YF constraint, to be changed in case other source is used
    """
    revenue_growth, eps_growth, fcf_growth = 0, 0, 0
    
    if len(income_df) >= years:
        revenue_growth = ((income_df['Total Revenue'].iloc[0] /income_df['Total Revenue'].iloc[years])**(1/years) - 1)
        eps_growth = calc_pos_neg_metrics(income_df, 'Basic EPS')
    if len(fcf_df) >= years:
        fcf_growth = calc_pos_neg_metrics(fcf_df, 'Free Cash Flow')
    
    return revenue_growth, eps_growth, fcf_growth

def calc_margin_measures(income_df, years = 3):
    """
    As of now based on three years due to YF constraint, to be changed in case other source is used
    """
    gross_margin_avg, net_margin_avg = 0, 0
    if len(income_df) >= years:
        if 'Gross Profit' in income_df.columns:
            gross_margin = income_df['Gross Profit'] / income_df['Total Revenue']
        else:
            # Mainly applied for banks
            gross_margin = income_df['Net Income'] / income_df['Total Revenue']
        net_margin       = income_df['Net Income'] / income_df['Total Revenue']
        gross_margin_avg = gross_margin.iloc[:years].mean()
        net_margin_avg   = net_margin.iloc[:years].mean()
    
    return gross_margin_avg, net_margin_avg

def calc_balance_measures(balance_df):
    """
    As of now based on three years due to YF constraint, to be changed in case other source is used
    """
    d_e_ratio, current_ratio, quick_ratio = 0, 0, 0
    if len(balance_df) >= 1:
        total_debt   = balance_df.get('Total Debt', 0).iloc[0]
        total_equity = balance_df.get('Stockholders Equity', 0).iloc[0]
        d_e_ratio    = total_debt / total_equity if total_equity != 0 else 0
        
        if 'Current Assets' in balance_df.columns:
            current_ratio = balance_df.get('Current Assets', 0).iloc[0] / balance_df.get('Current Liabilities', 0).iloc[0]
            if 'Inventory' in balance_df.columns and balance_df['Inventory'].iloc[0] > 0:
                quick_ratio   = (balance_df.get('Current Assets', 0).iloc[0] - balance_df.get('Inventory', 0).iloc[0]) / balance_df.get('Current Liabilities', 0).iloc[0]
            else:
                quick_ratio = current_ratio
        else:
            # !!! PROXY, MAINLY FOR BANKS
            # TO BE REVIEWED IF OK
            current_ratio = balance_df.get('Cash And Cash Equivalents', 0).iloc[0] / balance_df.get('Current Debt', 0).iloc[0]
            quick_ratio = current_ratio
        
        current_ratio = current_ratio
        quick_ratio   = quick_ratio
    
    return -d_e_ratio, current_ratio, quick_ratio

def calc_altman_z(income_df, balance_df, market_cap):
    """
    As of now based on three years due to YF constraint, to be changed in case other source is used
    """
    if 'Working Capital' in balance_df.columns:
        A = balance_df.get('Working Capital', 0).iloc[0] / balance_df.get('Total Assets', 0).iloc[0]
    else:
        # in case of banks assume Working Capital = Cash - Current Debt
        A = (balance_df.get('Cash And Cash Equivalents', 0).iloc[0] - balance_df.get('Current Debt', 0).iloc[0]) / balance_df.get('Total Assets', 0).iloc[0]
    B = balance_df.get('Retained Earnings', 0).iloc[0] / balance_df.get('Total Assets', 0).iloc[0]
    if 'EBIT' in income_df.columns:
        C = income_df.get('EBIT', 0).iloc[0] / balance_df.get('Total Assets', 0).iloc[0]
    else:
        # in case of banks assume Net Income
        C = income_df.get('Net Income', 0).iloc[0] / balance_df.get('Total Assets', 0).iloc[0]
    D = market_cap / balance_df.get('Total Liabilities Net Minority Interest', 0).iloc[0]
    E = income_df.get('Total Revenue', 0).iloc[0] / balance_df.get('Total Assets', 0).iloc[0]
    Z_score = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
    
    return Z_score

def calc_no_y_pos_msr(df, metric):
    return sum(df[metric] > 0)

def calc_shares_change(df, metric):
    return (df[metric].dropna().iloc[0] / df[metric].dropna().iloc[-1] - 1)

def retrieve_info_data(info, do_quant):
    """
    Based on info structure in YF
    Takes into account situations, when some attributes are not present
    In case no option is available, assigns very low value penalizing the stock
    """

    if do_quant:
        marketCap = info['marketCap']
        pb_ratio = -info['priceToBook']
        ps_ratio = -info['priceToSalesTrailing12Months']
        if 'forwardPE' in info.keys():
            pe_ratio = -info['forwardPE']
        elif 'trailingPE' in info.keys():
            pe_ratio = -info['trailingPE']
        else:
            pe_ratio = -100
            
        if 'enterpriseToEbitda' in info.keys():
            ev_ebitda = -info['enterpriseToEbitda']
        elif 'enterpriseToRevenue' in info.keys():
            ev_ebitda = -info['enterpriseToRevenue']
        else:
            ev_ebitda = -100
        # Extract beta --> negative since, in this case, it is interpreted as the higher the worst (prefer lower vol)
        beta = -info['beta']
    # different structure of input in case pulled from excel
    else:
        marketCap = info['marketCap'][0]
        pb_ratio = -info['priceToBook'][0]
        ps_ratio = -info['priceToSalesTrailing12Months'][0]
        if 'forwardPE' in info.keys():
            pe_ratio = -info['forwardPE'][0]
        elif 'trailingPE' in info.keys():
            pe_ratio = -info['trailingPE'][0]
        else:
            pe_ratio = -100
            
        if 'enterpriseToEbitda' in info.keys():
            ev_ebitda = -info['enterpriseToEbitda'][0]
        elif 'enterpriseToRevenue' in info.keys():
            ev_ebitda = -info['enterpriseToRevenue'][0]
        else:
            ev_ebitda = -100
        beta = -info['beta'][0]
            
    return marketCap, pb_ratio, ps_ratio, pe_ratio, ev_ebitda, beta

def normalize_dataframe(df, min_max_cols, z_score_cols):
    """
    Normalizes specified columns in a pandas DataFrame using Min-Max and Z-score scaling.
    
    Args:
        df (pd.DataFrame): The DataFrame to normalize.
        min_max_cols (list): A list of column names to normalize using Min-Max scaling.
                             Best for metrics with only positive values (e.g., margins, ratios).
        z_score_cols (list): A list of column names to normalize using Z-score normalization.
                             Best for metrics with positive and negative values (e.g., growth rates).
                             
    Returns:
        pd.DataFrame: A new DataFrame with the specified columns normalized.
    """
    df_normalized = df.copy()
    
    # Min-Max Scaling
    for col in min_max_cols:
        if col in df_normalized.columns:
            # Handle the case where max and min are the same to avoid division by zero
            if df_normalized[col].max() == df_normalized[col].min():
                df_normalized[col] = 0
            else:
                df_normalized[col] = (df_normalized[col] - df_normalized[col].min()) / \
                                      (df_normalized[col].max() - df_normalized[col].min())
    
    # Z-Score Normalization
    for col in z_score_cols:
        if col in df_normalized.columns:
            mean = df_normalized[col].mean()
            std_dev = df_normalized[col].std()
            # Handle the case where standard deviation is zero to avoid division by zero
            if std_dev == 0:
                df_normalized[col] = 0
            else:
                df_normalized[col] = (df_normalized[col] - mean) / std_dev
                
    return df_normalized
################################################# QUAL ###########################################################
class CompanyAnalyzer:
    def __init__(self, companies, income_df, balance_df, cash_flow_df, 
                      quant_moat_params, quant_moat_scores, moat_keywords, 
                      leadership_keywords, keyword_score_weight, sentiment_score_weight, qual_scorecard_weights):
        """
        Initializes the analyzer with a list of companies.
        Each company is a dictionary with a 'ticker' and 'website'.
        """
        self.companies = companies
        self.sia     = SentimentIntensityAnalyzer()
        self.results = []
        # Pre-compiled regex for efficiency
        self.moat_keywords = re.compile(
            moat_keywords,
            re.IGNORECASE
        )
        self.leadership_keywords = re.compile(
            leadership_keywords,
            re.IGNORECASE
        )
        self.income_df    = income_df
        self.balance_df   = balance_df
        self.cash_flow_df = cash_flow_df
        
        self.quant_moat_params = quant_moat_params
        self.quant_moat_scores = quant_moat_scores
        
        self.keyword_score_weight = keyword_score_weight
        self.sentiment_score_weight = sentiment_score_weight
        
        self.qual_scorecard_weights = qual_scorecard_weights

    def scrape_website_text(self, url):
        """
        Scrapes a given URL and returns the text content.
        Simple web scraping, can be enhanced for dynamic sites with Selenium.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        }
        try:
            response = requests.get(url, timeout=20, headers = headers)
            soup     = BeautifulSoup(response.content, 'html.parser')
            # Extract text from common tags, excluding script and style tags
            text     = ' '.join(t.get_text() for t in soup.find_all(['p', 'h1', 'h2', 'h3']))
            return text
        except requests.exceptions.RequestException as e:
            print(f"Error scraping {url}: {e}")
            return ""

    def analyze_moat_from_text(self, text):
        """
        Analyzes scraped text for moat-related keywords.
        Returns a count of matches.
        """
        # Store the MOAT words found in qual folder
        df = pd.DataFrame(self.moat_keywords.findall(text))
        df.to_csv(str("output\stock_qual_scores" + "\\" + self.companies[0]['ticker'] + "_moat_keywords.csv"))
        return len(self.moat_keywords.findall(text))
    
    def preprocess_text(self, text):
        """
        Preprocess the earnings call text to better capture pos/neg sentiment
        """
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
    
    def analyze_management_strength(self, text):
        """
        Analyzes text for management-related keywords and sentiment.
        Returns a score based on keyword presence and a basic sentiment score.
        """
        keyword_score   = len(self.leadership_keywords.findall(text))
        # Store the leadership words found in qual folder
        df = pd.DataFrame(self.leadership_keywords.findall(text))
        df.to_csv(str("output\stock_qual_scores" + "\\" + self.companies[0]['ticker'] + "_leader_keywords.csv"))
        
        return keyword_score * self.keyword_score_weight

    def analyze_sentiment_strength(self, text):
        """
        Analyzes sentiment with nltk Sentiment Analyzer
        Returns a score based on a difference of positive and negative sentiment times the predefined wieght.
        """
        # Preprocess the text
        preprocess_text = self.preprocess_text(text)
        # Take the score as a difference between positive and negative
        sentiment_score = self.sia.polarity_scores(preprocess_text)['pos'] - self.sia.polarity_scores(preprocess_text)['neg']
        # Store the sentiment score found in qual folder
        df_sent = pd.DataFrame([sentiment_score])
        df_sent.to_csv(str("output\stock_qual_scores" + "\\" + self.companies[0]['ticker'] + "_sentiment_score.csv"))
        
        return sentiment_score * self.sentiment_score_weight  # Arbitrary weighting for sentiment

    
    def run_analysis(self):
        """
        Main method to run the analysis for all companies.
        """
        for company in self.companies:
            ticker  = company['ticker']
            website = company['website']
            
            print(f"Analyzing {ticker}...")
            
            # Acquire data from company website
            full_text = self.scrape_website_text(website)
            if not full_text:
                print(f"Skipping {ticker} due to scraping error.")
                continue

            # Calculate MOAT strength (text-based part)
            moat_text_score = self.analyze_moat_from_text(full_text)

            # Pull financial data here and calculate quant scores
            quant_moat_score = self.calculate_quant_moat(ticker, self.income_df, self.balance_df)
            
            # Calculate management stability
            management_score = self.analyze_management_strength(full_text)
            
            # Calculate sentiment score
            sentiment_score = self.analyze_sentiment_strength(full_text)
            
            # Create the scorecard
            final_score = self.create_scorecard(ticker, moat_text_score, quant_moat_score, management_score, sentiment_score)
            
            self.results.append({
                'Ticker': ticker,
                'Moat_Text_Score' : moat_text_score,
                'Moat_Quant_Score': quant_moat_score,
                'Management_Score': management_score,
                'Sentiment_Score': sentiment_score,
                'Total_Qualitative_Score': final_score
            })
        
        # Display results in a DataFrame
        df = pd.DataFrame(self.results)
        print("\n--- Qualitative Scorecard Results ---")
        print(df)
        
        # Store as csv
        df.to_csv(str("output\stock_qual_scores" + "\\" + ticker + "_qual_scores.csv"))
        
    def calculate_quant_moat(self, ticker, income_df, balance_df):
        """
        This is a placeholder for your FMP data integration.
        You would connect to the FMP API here, pull data like margins and ROIC,
        and apply a scoring logic.
        """
        # Example scoring logic:
        # Assuming you pull a gross margin and ROIC
        # Let's say high margin (>50%) gets 5 pts, good ROIC (>15%) gets 5 pts.
        # This is where you would define your quantitative rules.
        if 'Gross Profit' in income_df[ticker].columns:
            gross_margin = income_df[ticker]['Gross Profit'].iloc[0] / income_df[ticker]['Total Revenue'].iloc[0]
        else:
            gross_margin = income_df[ticker]['Net Income'].iloc[0] / income_df[ticker]['Total Revenue'].iloc[0]
            
        roic = income_df[ticker]['Net Income'].iloc[0] / balance_df[ticker]['Stockholders Equity'].iloc[0]
        
        score = 0
        # !!!! MAYBE ADD ADDITIONAL POINTS IF LOW STD OF GROSS MARGIN / ROIC
        if gross_margin > self.quant_moat_params['Gross margin']:
            score += self.quant_moat_scores['Gross margin']
        if roic > self.quant_moat_params['ROIC']:
            score += self.quant_moat_scores['ROIC']
        
        return score

    def create_scorecard(self, ticker, moat_text, moat_quant, management, sentiment):
        """
        Calculates a final composite score based on the individual scores.
        You can customize the weighting of each factor here.
        """
        # Define weights for each factor
        WEIGHT_MOAT_TEXT  = self.qual_scorecard_weights['MOAT_TEXT']
        WEIGHT_MOAT_QUANT = self.qual_scorecard_weights['MOAT_QUANT']
        WEIGHT_MANAGEMENT = self.qual_scorecard_weights['MANAGEMENT']
        WEIGHT_SENTIMENT = self.qual_scorecard_weights['SENTIMENT']
        
        total_score = (moat_text  * WEIGHT_MOAT_TEXT  + 
                       moat_quant * WEIGHT_MOAT_QUANT + 
                       management * WEIGHT_MANAGEMENT + 
                       sentiment  * WEIGHT_SENTIMENT)
        
        return total_score

def plot_ranks(df, metric):
    """
    Plot bar charts and store as png with sorted main metrics and ranks
    """
    # Create a sample DataFrame
    index_labels = df.index
    df = df.sort_values(by = metric)

    # Create the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(df.index, df[metric], color='skyblue')

    # Add labels and a title
    plt.xlabel('Ticker')
    plt.ylabel(metric)
    plt.title(str('Sorted ' + metric))

    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation = 45, ha='right')

    # Add a tight layout to prevent labels from being cut off
    plt.tight_layout()
    # Save the plot
    plt.savefig(str("output\\figures" + "\\" + "_bar_chart_" + metric + ".png"))
    plt.show()

def create_pdf_report(image_folder, output_pdf):
    """
    Combines PNG plots into a PDF with automatic descriptions.
    """
    # Create the PDF document
    doc = SimpleDocTemplate(output_pdf, pagesize=letter)
    
    # Story is a list of flowables (like paragraphs, images, spacers) that will be added to the PDF
    Story = []
    
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading2']
    body_style = styles['Normal']
    body_style.wordWrap = 'CJK'
    
    # Add a title page
    Story.append(Paragraph("Stock Performance Report", title_style))
    Story.append(Spacer(1, 0.5*inch))
    
    # Get a list of all PNG files in the folder and sort them
    image_files = ["_bar_chart_Final Rank.png", 
                   "_bar_chart_Quant Rank.png", "_bar_chart_revenue_growth_rank.png", "_bar_chart_eps_growth_rank.png",
                   "_bar_chart_gross_margin_avg_rank.png", "_bar_chart_net_margin_avg_rank.png", 
                   "_bar_chart_fcf_growth_rank.png", "_bar_chart_d_e_ratio_rank.png", 
                   "_bar_chart_curr_ratio_rank.png", "_bar_chart_quick_ratio_rank.png",
                   "_bar_chart_z_score_rank.png", "_bar_chart_no_y_pos_eps_rank.png", 
                   "_bar_chart_no_y_pos_fcf_rank.png", "_bar_chart_shares_chg_rank.png",
                   "_bar_chart_pe_ratio_rank.png", "_bar_chart_ps_ratio_rank.png",
                   "_bar_chart_pb_ratio_rank.png", "_bar_chart_ev_ebitda_rank.png", 
                   "_bar_chart_beta_rank.png", "_bar_chart_scaled_vol_rank.png",
                   "_bar_chart_accrual_rat_rank.png", "_bar_chart_cash_conv_rat_rank.png",
                   "_bar_chart_Qual Rank.png", "_bar_chart_MOAT Quant Score.png", "_bar_chart_MOAT Text Score.png", "_bar_chart_Management Score.png", "_bar_chart_Sentiment Score.png",
                   "_bar_chart_External Rating Score.png"]

    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        # Force a page break
        Story.append(PageBreak())
        # Add a heading for the current plot
        Story.append(Paragraph(f"Analysis of {img_file}", heading_style))
        Story.append(Spacer(1, 0.2*inch))
        
        # Add the plot image to the PDF
        img = RLImage(img_path)
        img.drawHeight = 4*inch
        img.drawWidth = 6*inch
        Story.append(img)
        Story.append(Spacer(1, 0.2*inch))
        
        # Generate and add the description
        description = ""
        Story.append(Paragraph(description, body_style))
        Story.append(Spacer(1, 0.5*inch))
        Story.append(Paragraph("-" * 50, body_style)) # Add a separator
        
    # Build the PDF document
    doc.build(Story)
    print(f"PDF report '{output_pdf}' created successfully!")
