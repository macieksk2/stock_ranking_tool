# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 20:09:58 2025

@author: macie
"""
import os
import pandas as pd
import numpy as np
import requests
import re

import yfinance as yf
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer

# import relevant functions
from utils.utils import *
from utils.utils_parse_Cymcyk import *
# import input parameters
from input.params import *
############################################################## INPUT #################################################
# Master dictionary to store all scores for all tickers
master_scores = {}
metrics_data = {ticker: {} for ticker in tickers}

if __name__ == '__main__':
    ############################################### QUANTITATIVE PART #######################################
    print('*************************************************')
    print('QUANTITATIVE PART')
    if DO_PULL_QUANT:
        income_df, balance_df, cash_flow_df, market_caps = create_df_yf(tickers)
        for ticker in tickers:
            income_df[ticker].to_excel(str("output\stock_quant_scores"    + "\\" + ticker + "_income_df.xlsx"))
            balance_df[ticker].to_excel(str("output\stock_quant_scores"   + "\\" + ticker + "_balance_df.xlsx"))
            cash_flow_df[ticker].to_excel(str("output\stock_quant_scores" + "\\" + ticker + "_cash_flow_df.xlsx"))
            pd.DataFrame([market_caps[ticker]]).to_excel(str("output\stock_quant_scores" + "\\" + ticker + "_market_cap.xlsx"))
    # False - feed file by file from excels
    else:
        income_df, balance_df, cash_flow_df, market_caps = dict(), dict(), dict(), dict()
        for ticker in tickers:
            income_df[ticker]    = pd.read_excel(str("output\stock_quant_scores" + "\\" + ticker + "_income_df.xlsx"))
            balance_df[ticker]   = pd.read_excel(str("output\stock_quant_scores" + "\\" + ticker + "_balance_df.xlsx"))
            cash_flow_df[ticker] = pd.read_excel(str("output\stock_quant_scores" + "\\" + ticker + "_cash_flow_df.xlsx"))
            market_caps[ticker]  = pd.read_excel(str("output\stock_quant_scores" + "\\" + ticker + "_market_cap.xlsx"))
    
    for ticker in tickers:
        print('*****************************')
        print(ticker)
        # --- Quantitative Metric Calculation ---
        # Growth measures
        revenue_growth, eps_growth, fcf_growth = calc_income_fcf_metrics(income_df[ticker], cash_flow_df[ticker], years = 3)
        # Margin measures
        gross_margin_avg, net_margin_avg       = calc_margin_measures(income_df[ticker], years = 3)
        # Balance sheet measures
        # D/E set to negative since the lower the better
        d_e_ratio, current_ratio, quick_ratio  = calc_balance_measures(balance_df[ticker])
        # Altman Z-score to measure risk of company default
        z_score = calc_altman_z(income_df[ticker], balance_df[ticker], market_caps[ticker][0][0])
        # Number of years with positive EPS and FCF
        no_y_pos_eps, no_y_pos_fcf = calc_no_y_pos_msr(income_df[ticker], 'Basic EPS'), calc_no_y_pos_msr(cash_flow_df[ticker], 'Free Cash Flow')
        # %change in shares outstanding (negative since the largest decrease the better and vice versa)
        shares_chg = -calc_shares_change(income_df[ticker], 'Basic Average Shares')
        # Store the raw quantitative metrics
        metrics_data[ticker] = {
            'revenue_growth'  : revenue_growth,
            'eps_growth'      : eps_growth,
            'gross_margin_avg': gross_margin_avg,
            'net_margin_avg'  : net_margin_avg,
            'fcf_growth'      : fcf_growth,
            'd_e_ratio'       : d_e_ratio,
            'curr_ratio'      : current_ratio,
            'quick_ratio'     : quick_ratio,
            'z_score'         : z_score,
            'no_y_pos_eps'    : no_y_pos_eps,
            'no_y_pos_fcf'    : no_y_pos_fcf,
            'shares_chg'      : shares_chg
        }
    ################################################# QUALITATIVE PART #######################################
    print('*************************************************')
    print('QUALITATIVE PART')
    # If False, it should read the score results from csv
    if DO_PULL_QUAL:
        for ticker in tickers:
            # Define the list of companies to analyze
            # This list would be generated from your FMP data or a separate input file
            companies_to_analyze = [{'ticker': ticker, 'website': inv_websites[ticker]}]
        
            analyzer = CompanyAnalyzer(companies_to_analyze, income_df, balance_df, cash_flow_df, 
                                       quant_moat_params, quant_moat_scores, moat_keywords, 
                                       leadership_keywords, keyword_score_weight, sentiment_score_weight, qual_scorecard_weights)
            analyzer.run_analysis()
                        
            master_scores[ticker] = {'MOAT Text Score' :  analyzer.results[0]['Moat_Text_Score'],
                                     'MOAT Quant Score':  analyzer.results[0]['Moat_Quant_Score'],
                                     'Management Score':  analyzer.results[0]['Management_Score'],
                                     'Sentiment Score' :  analyzer.results[0]['Sentiment_Score'],
                                     'Qual Score'      :  analyzer.results[0]['Total_Qualitative_Score']}
    else:
        for ticker in tickers:
            qual_score = pd.read_csv(str("output\stock_qual_scores" + "\\" + ticker + "_qual_scores.csv"))
            master_scores[ticker] = {'MOAT Text Score'  :  qual_score['Moat_Text_Score'][0],
                                     'MOAT Quant Score' :  qual_score['Moat_Quant_Score'][0],
                                     'Management Score' :  qual_score['Management_Score'][0],
                                     'Sentiment Score'  :  qual_score['Sentiment_Score'][0],
                                     'Qual Score'       :  qual_score['Total_Qualitative_Score'][0]}
            
    ############## QUALITATIVE PART FROM CYMCYK ###################################################################
    print('*************************************************')
    print('EXTERNAL RATING PART')
    if DO_EXT_RATING:
        external_ratings = get_analyst_rankings_from_pdf(pdf_path, tickers, ticker_mapping)
        for ticker in tickers:
            master_scores[ticker]['External Rating'] = int(external_ratings[0][ticker])
    
    ############## RANK CALCULATION ###############################################################################
    # Rank the quantitative metrics across all tickers ---
    metrics_df = pd.DataFrame(metrics_data).transpose()
    
    # Calculate the weighted average of the scores for the quantitative score
    metrics_df['Quant Score'] = 0  
    for metric, weight in quant_weights.items():
        metrics_df['Quant Score'] += metrics_df[metric] * weight
        
    # Rank each quantitative column. For ratios like D/E, a lower value is better, so ascending=True
    ranked_metrics = metrics_df.copy()
    for metric in quant_weights.keys():
        ranked_metrics[metric + '_rank'] = metrics_df[metric].rank(ascending=False)
    ranked_metrics['Quant Score'] = metrics_df['Quant Score']    
    ranked_metrics['Quant Score_rank'] = metrics_df['Quant Score'].rank(ascending=False)
    # Combine all scores into a final DataFrame
    final_df                      = ranked_metrics.copy()
    final_df['MOAT Text Score']   = pd.Series({t: master_scores[t]['MOAT Text Score'] for t in tickers})
    final_df['MOAT Quant Score']  = pd.Series({t: master_scores[t]['MOAT Quant Score'] for t in tickers})
    final_df['Management Score']  = pd.Series({t: master_scores[t]['Management Score'] for t in tickers})
    final_df['Sentiment Score']   = pd.Series({t: master_scores[t]['Sentiment Score'] for t in tickers})
    final_df['Qual Score']  = pd.Series({t: master_scores[t]['Qual Score'] for t in tickers})
    if DO_EXT_RATING:
        final_df['External Rating Score'] = pd.Series({t: master_scores[t]['External Rating'] for t in tickers})
    # Rank the final scores
    final_df['Quant Rank'] = final_df['Quant Score_rank']
    final_df['Qual Rank']  = final_df['Qual Score'].rank(ascending=False)
    final_df['Final Rank'] = (final_df['Quant Rank'] + final_df['Qual Rank']) / 2
    # If specified, include a rank of external rating into the composite rank
    if DO_EXT_RATING:
        final_df['External Rating Rank']  = final_df['External Rating Score'].rank(ascending=False)
        final_df['Final Rank'] = (final_df['Quant Rank'] + final_df['Qual Rank'] + final_df['External Rating Rank']) / 3
    if not DO_EXT_RATING:
        print(final_df[['Quant Score', 'Qual Score', 'Quant Rank', 'Qual Rank', 'Final Rank']])
    else:
        print(final_df[['Quant Score', 'Qual Score', 'Quant Rank', 'Qual Rank', 'External Rating Rank', 'Final Rank']])
    final_df.to_excel("output\stock_analysis_scores.xlsx")
    
    ############## PLOTTING ########################################################################################
    for metric in final_df.columns:
        plot_ranks(final_df, metric)
    # Create a pdf report with plots
    create_pdf_report("output\\figures", "output\\Stock_Performance_Report.pdf")

    
