# -*- coding: utf-8 -*-
"""
input from the user
"""

# Replace with your actual API key from Financial Modeling Prep
# Currently not used sicne data from YF
API_KEY = "O7jpbikr5wCjo34KFLWi9nmU3xJ9laxh"

# The tickers and weights for the quantitative and qualitative factors
tickers = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'TSLA', 
           'NVDA', 'META', 'ASML', 'AMD', 'ARM', 'JPM', 'GS', 'INTU',
           'TSMC34.SA', 'RACE', 'PLTR', 'LLY',
           'VRTX', 'KMI', 'C', 'INTC', 'BABA', 'RIO', 'EQR', 'LMT', 'BATS.L']#, 'HIMS']

# Set flags whether the data should be pulled online (True) or from local files (False)
DO_PULL_QUANT = False
DO_PULL_QUAL  = False
# True if parse rating from Cymcyk analysis
DO_EXT_RATING = True

############################################################ QUANTITATIVE ############################################
# Weights for the quantitative score (ensure they sum to 1.0)
# !!!! THINK ABOUT THE WEIGHTING
quant_weights = {
    'revenue_growth':    0.15,
    # not enough data for 10Y in YF
    # 'revenue_growth_10y': 0.10,
    'eps_growth':        0.15,
    # 'eps_growth_10y': 0.075,
    'gross_margin_avg':  0.1,
    'net_margin_avg':    0.1,
    'fcf_growth':        0.05,
    'd_e_ratio':         0.075,
    'curr_ratio':        0.075,
    'quick_ratio':       0.05,
    'z_score':           0.10,
    'no_y_pos_eps':      0.05,
    'no_y_pos_fcf':      0.05,
    'shares_chg':        0.05
}

############################################################ QUALITATIVE ############################################
# Quantitative MOAT score params (minimum gross margin and ROIC)
quant_moat_params = {
    'Gross margin': 0.4,
    'ROIC': 0.15
    }
# Scores related to reaching Quant MOAT measures
quant_moat_scores = {
    'Gross margin': 5,
    'ROIC': 5
    }

# !!! RETHINK THE WORDS THAT MAY ACTUALLY INDICATE MOAT OR LEADERSHIP STRENGTH
# !!! ADD POLISH COUNTERPARTS
# MOAT keywords
moat_keywords = r'patent|trademark|brand|loyalty|network effect|switching costs|economies of scale|scale benefits|proprietary technology|regulatory advantage|most successful|the best source|best-in-class'
# Leadership network keywords
leadership_keywords = r'innovative|innovation|innovations|experienced|veteran|visionary|strategic|leadership|board of directors|revolutionary|revolutionized|unmatched|massive scale'

# Management strength parametrization
keyword_score_weight = 1 
sentiment_score_weight = 40

# Qualitiative scoreard weights
# !!! THINK ABOUT THE WEIGHTINGS
qual_scorecard_weights = {
    "WEIGHT_MOAT_TEXT"  : 0.3,
    "WEIGHT_MOAT_QUANT" : 0.4,
    "WEIGHT_MANAGEMENT" : 0.15,
    "WEIGHT_SENTIMENT" : 0.15   
    }

# Dictionary with companies vs links to investor websites
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

############################################################ EXTERNAL ############################################
# Input to parse pdf from Cymcyk analysis (map tickers to actual names used)
ticker_mapping = {'MSFT': 'Microsoft',
                  'AAPL': 'Apple',
                  'GOOGL': 'Alphabet',
                  'META': 'Meta Platform',
                  'AMZN': 'Amazon',
                  'NVDA': 'Nvidia',
                  'INTU': 'Intuitive Surgical',
                  'TSLA': 'Tesla',
                  'BABA': 'Alibaba',
                  'TSMC34.SA': 'TSMC',
                  'ASML': 'ASML',
                  'RACE': 'Ferrari',
                  'PLTR': 'Palantir',
                  'LLY' : 'Eli Lilly',
                  'C': 'Citi'}

pdf_path = r'input\StockScan_grudzien_2024_fin-laqvwb.pdf'