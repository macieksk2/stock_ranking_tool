# -*- coding: utf-8 -*-
"""
main file of Stock Scorecard framework
"""
import os
import pandas as pd
import numpy as np
import requests
import re
import yaml
from yaml.loader import SafeLoader

import yfinance as yf
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer

# import relevant functions
from utils.utils import *
from utils.utils_parse_Cymcyk import *
# import input parameters
# Load input parameters from YAML file
with open('input/params.yaml', 'r') as f:
    config = yaml.load(f, Loader=SafeLoader)

############################################################## INPUT #################################################
tickers = config['tickers']

# Master dictionary to store all scores for all tickers
master_scores = {}
metrics_data = {ticker: {} for ticker in tickers}

if __name__ == '__main__':
    ############################################### QUANTITATIVE PART #######################################
    print('*************************************************')
    print('QUANTITATIVE PART')
    print('*************************************************')
    if config['use_quant']:
        income_df, balance_df, cash_flow_df, info = create_df_yf(tickers)
        for ticker in tickers:
            income_df[ticker].to_excel(str("output\stock_quant_scores"    + "\\" + ticker + "_income_df.xlsx"))
            balance_df[ticker].to_excel(str("output\stock_quant_scores"   + "\\" + ticker + "_balance_df.xlsx"))
            cash_flow_df[ticker].to_excel(str("output\stock_quant_scores" + "\\" + ticker + "_cash_flow_df.xlsx"))
            pd.DataFrame([info[ticker]]).to_excel(str("output\stock_quant_scores" + "\\" + ticker + "_info.xlsx"))
    # False - feed file by file from excels
    else:
        income_df, balance_df, cash_flow_df, info = dict(), dict(), dict(), dict()
        for ticker in tickers:
            income_df[ticker]    = pd.read_excel(str("output\stock_quant_scores" + "\\" + ticker + "_income_df.xlsx"))
            balance_df[ticker]   = pd.read_excel(str("output\stock_quant_scores" + "\\" + ticker + "_balance_df.xlsx"))
            cash_flow_df[ticker] = pd.read_excel(str("output\stock_quant_scores" + "\\" + ticker + "_cash_flow_df.xlsx"))
            info[ticker]         = pd.read_excel(str("output\stock_quant_scores" + "\\" + ticker + "_info.xlsx"))
    
    for ticker in tickers:
        print('*****************************')
        print(ticker)
        print('*************************************************')
        # --- Quantitative Metric Calculation ---
        # Growth measures
        revenue_growth, eps_growth, fcf_growth = calc_income_fcf_metrics(income_df[ticker], cash_flow_df[ticker], years = 3)
        # Margin measures
        gross_margin_avg, net_margin_avg  = calc_margin_measures(income_df[ticker], years = 3)
        # Balance sheet measures
        # D/E set to negative since the lower the better
        d_e_ratio, current_ratio, quick_ratio  = calc_balance_measures(balance_df[ticker])
        # Number of years with positive EPS and FCF
        no_y_pos_eps, no_y_pos_fcf = calc_no_y_pos_msr(income_df[ticker], 'Basic EPS'), calc_no_y_pos_msr(cash_flow_df[ticker], 'Free Cash Flow')
        # %change in shares outstanding (negative since the largest decrease the better and vice versa)
        shares_chg = -calc_shares_change(income_df[ticker], 'Basic Average Shares')
        # Pull metrics from infos, based on type of input (online or from excel)
        if config['use_quant']:
            marketCap = info[ticker]['marketCap']
            pb_ratio = -info[ticker]['priceToBook']
            ps_ratio = -info[ticker]['priceToSalesTrailing12Months']
            if 'forwardPE' in info[ticker].keys():
                pe_ratio = -info[ticker]['forwardPE']
            elif 'trailingPE' in info[ticker].keys():
                pe_ratio = -info[ticker]['trailingPE']
            else:
                pe_ratio = -100
                
            if 'enterpriseToEbitda' in info[ticker].keys():
                ev_ebitda = -info[ticker]['enterpriseToEbitda']
            elif 'enterpriseToRevenue' in info[ticker].keys():
                ev_ebitda = -info[ticker]['enterpriseToRevenue']
            else:
                ev_ebitda = -100
            # Extract beta --> negative since, in this case, it is interpreted as the higher the worst (prefer lower vol)
            beta = -100 * info[ticker]['beta']
        else:
            marketCap = info[ticker]['marketCap'][0]
            pb_ratio = -info[ticker]['priceToBook'][0]
            ps_ratio = -info[ticker]['priceToSalesTrailing12Months'][0]
            if 'forwardPE' in info[ticker].keys():
                pe_ratio = -info[ticker]['forwardPE'][0]
            elif 'trailingPE' in info[ticker].keys():
                pe_ratio = -info[ticker]['trailingPE'][0]
            else:
                pe_ratio = -100
                
            if 'enterpriseToEbitda' in info[ticker].keys():
                ev_ebitda = -info[ticker]['enterpriseToEbitda'][0]
            elif 'enterpriseToRevenue' in info[ticker].keys():
                ev_ebitda = -info[ticker]['enterpriseToRevenue'][0]
            else:
                ev_ebitda = -100
            # Extract beta --> negative since, in this case, it is interpreted as the higher the worst (prefer lower vol)
            beta = -100 * info[ticker]['beta'][0]
        
        # Calculate scaled volatility of eps --> negative since the higher the worst
        scaled_vol = -100 * np.std(income_df[ticker]['Basic EPS']) / np.mean(abs(income_df[ticker]['Basic EPS']))
    
        # Altman Z-score to measure risk of company default
        z_score = calc_altman_z(income_df[ticker], balance_df[ticker], marketCap)
        # Accrual ratio (earnings vs cash flow) --> negative (high income not backed by cash is a red flag)
        accrual_rat = -100 * (income_df[ticker]['Net Income'].iloc[0] - cash_flow_df[ticker]['Operating Cash Flow'].iloc[0]) / balance_df[ticker]['Total Assets'].iloc[0]
        # Cash Conversion Ratio
        cash_conv_rat = 10 * cash_flow_df[ticker]['Operating Cash Flow'].iloc[0] / income_df[ticker]['Net Income'].iloc[0] 
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
            'shares_chg'      : shares_chg,
            'pe_ratio'        : pe_ratio, 
            'ps_ratio'        : ps_ratio, 
            'pb_ratio'        : pb_ratio,
            'ev_ebitda'       : ev_ebitda,
            'beta'            : beta,
            'scaled_vol'      : scaled_vol,
            'accrual_rat'     : accrual_rat,
            'cash_conv_rat'   : cash_conv_rat
        }
    ################################################# QUALITATIVE PART #######################################
    print('*************************************************')
    print('QUALITATIVE PART')
    print('*************************************************')
    # If False, it should read the score results from csv
    if config['use_qual']:
        for ticker in tickers:
            # Define the list of companies to analyze
            # This list would be generated from your FMP data or a separate input file
            companies_to_analyze = [{'ticker': ticker, 'website': config['transcript_urls'][ticker]}]
        
            analyzer = CompanyAnalyzer(companies_to_analyze, income_df, balance_df, cash_flow_df, 
                                       config['quant_moat_params'], config['quant_moat_scores'], config['moat_keywords'], 
                                       config['leadership_keywords'], config['keyword_score_weight'], config['sentiment_score_weight'], 
                                       config['qual_scorecard_weights'])
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
    print('*************************************************')
    if config['use_external']:
        external_ratings = get_analyst_rankings_from_pdf(config['pdf_path'], tickers, config['ticker_mapping'])
        for ticker in tickers:
            master_scores[ticker]['External Rating'] = int(external_ratings[0][ticker])
    
    ############## RANK CALCULATION ###############################################################################
    # Rank the quantitative metrics across all tickers ---
    metrics_df = pd.DataFrame(metrics_data).transpose()
    
    # Calculate the weighted average of the scores for the quantitative score
    metrics_df['Quant Score'] = 0  
    for metric, weight in config['quant_weights'].items():
        metrics_df['Quant Score'] += metrics_df[metric] * weight
        
    # Rank each quantitative column. For ratios like D/E, a lower value is better, so ascending=True
    ranked_metrics = metrics_df.copy()
    for metric in config['quant_weights'].keys():
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
    if config['use_external']:
        final_df['External Rating Score'] = pd.Series({t: master_scores[t]['External Rating'] for t in tickers})
    # Rank the final scores
    final_df['Quant Rank'] = final_df['Quant Score_rank']
    final_df['Qual Rank']  = final_df['Qual Score'].rank(ascending=False)
    final_df['Final Rank'] = (final_df['Quant Rank'] + final_df['Qual Rank']) / 2
    # If specified, include a rank of external rating into the composite rank
    if config['use_external']:
        final_df['External Rating Rank']  = final_df['External Rating Score'].rank(ascending=False)
        final_df['Final Rank'] = (final_df['Quant Rank'] + final_df['Qual Rank'] + final_df['External Rating Rank']) / 3
    if not config['use_external']:
        print(final_df[['Quant Score', 'Qual Score', 'Quant Rank', 'Qual Rank', 'Final Rank']])
    else:
        print(final_df[['Quant Score', 'Qual Score', 'Quant Rank', 'Qual Rank', 'External Rating Rank', 'Final Rank']])
    final_df.to_excel("output\stock_analysis_scores.xlsx")
    
    ############## PLOTTING ########################################################################################
    for metric in final_df.columns:
        plot_ranks(final_df, metric)
    # Create a pdf report with plots
    create_pdf_report("output\\figures", "output\\Stock_Performance_Report.pdf")

    
