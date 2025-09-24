# -*- coding: utf-8 -*-
"""
Created on Fri Sep 19 17:58:59 2025

@author: macie
"""

import pdfplumber
import pandas as pd
import re

def get_analyst_rankings_from_pdf(pdf_path, tickers, ticker_mapping):
    """
    Parses a PDF file to extract analyst rankings for a list of tickers.
    
    Args:
        pdf_path (str): The file path to the PDF document.
        tickers (list): A list of stock tickers to search for.
        
    Returns:
        pd.DataFrame: A DataFrame with tickers and their analyst ratings.
    """
    analyst_ratings = {}
    
    with pdfplumber.open(pdf_path) as pdf:
        # reverse pages since the summary slide is closer to the end of the deck
        for page in reversed(pdf.pages):
            print(page)
            text = page.extract_text()
            # Find relvenat headers in the summary slide
            match = text.find('Podsumowanie')
            match2 = text.find('ZAGRANICA')
            if match >= 0 and match2 >= 0:
                # search through each ticker to pull the rating
                # if not available, assign lowest (-2)
                for ticker in tickers:
                    print(ticker)
                    # Map the ticker with mapping; if not found, used the actual ticker
                    try:
                        map_ticker = ticker_mapping[ticker]
                    except:
                        map_ticker = ticker
                    start_index = text.find(map_ticker) - 2
                    # account for negative ratings
                    if text[start_index - 1] == "-":
                        start_index -= 1
                    end_index   = text.find(map_ticker) - 1
                    if start_index > 0:
                        rating = text[start_index:end_index]
                    else:
                        # default for not found is the worst rating, i.e. -2
                        rating = -2
                    analyst_ratings[ticker] = rating
                    print(rating)
                break
    
    # Create a DataFrame from the extracted ratings
    analyst_df = pd.DataFrame.from_dict(analyst_ratings, orient='index')
    analyst_df.index.name = 'Ticker'
    
    # Fill any missing tickers with the lowest ranking, as requested
    analyst_df = analyst_df.reindex(tickers)
    analyst_df.fillna(-2, inplace=True)
    
    return analyst_df

