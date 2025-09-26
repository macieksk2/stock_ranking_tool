# -*- coding: utf-8 -*-
"""
Unit tests for functions in utils.py using pytest.
"""
import pytest
import pandas as pd
import numpy as np
from collections import defaultdict

# Assuming utils.py is in the current directory or imported via setup
# For this example, we'll assume the functions are available:
from utils import (
    calc_income_fcf_metrics, 
    calc_margin_measures, 
    calc_balance_measures, 
    calc_altman_z, 
    calc_no_y_pos_msr, 
    calc_shares_change,
    calc_pos_neg_metrics,
    normalize_dataframe,
    retrieve_info_data
)

# --- Mock Data Setup ---

# Mock Income Statement (dates are indices, rows are metrics)
# @pytest.fixture
def mock_income_df_good():
    return pd.DataFrame({
        '2024-01-01': [1000, 100, 50, 20, 100, 100],  # Latest
        '2023-01-01': [800, 80, 40, 15, 80, 80],
        '2022-01-01': [600, 60, 30, 10, 60, 60],
        '2021-01-01': [500, 50, 25, 5, 50, 50]       # 3 years ago (index 3)
    }, index=['Total Revenue', 'Gross Profit', 'Net Income', 'Basic EPS', 'Basic Average Shares', 'EBIT']).transpose()

# Mock Cash Flow Statement
# @pytest.fixture
def mock_cashflow_df_good():
    return pd.DataFrame({
        '2024-01-01': [70, 60],
        '2023-01-01': [50, 40],
        '2022-01-01': [30, 20],
        '2021-01-01': [20, 10]
    }, index=['Operating Cash Flow', 'Free Cash Flow']).transpose()

# Mock Balance Sheet
# @pytest.fixture
def mock_balance_df_good():
    return pd.DataFrame({
        '2024-01-01': [1000, 200, 100, 50, 20, 10, 500, 500, 400, 500], # Latest
        '2023-01-01': [900, 180, 90, 45, 18, 9, 450, 450, 360, 450],
    }, index=['Total Assets', 'Current Assets', 'Current Liabilities', 'Inventory', 
              'Cash And Cash Equivalents', 'Current Debt', 'Total Debt', 
              'Stockholders Equity', 'Retained Earnings', 'Total Liabilities Net Minority Interest']).transpose()

# Mock Info Dictionary
# @pytest.fixture
def mock_info_dict():
    return {
        'marketCap': 1000000,
        'priceToBook': 5.0,
        'priceToSalesTrailing12Months': 2.5,
        'forwardPE': 20.0,
        'beta': 1.2,
        'enterpriseToEbitda': 15.0
    }

# --- Unit Tests for Quantitative Metrics ---

def test_calc_pos_neg_metrics_positive_growth():
    # Test normal positive growth
    df = pd.DataFrame({'Metric': [100, 80, 60, 50]}).rename(index=lambda x: str(x)).transpose()
    growth = calc_pos_neg_metrics(df.transpose(), 'Metric', years=3)
    # Expected: (100/50)^(1/3) - 1 ≈ 0.2599
    assert np.isclose(growth, 0.2599, rtol=1e-3)

def test_calc_pos_neg_metrics_neg_to_pos():
    # Test change from negative to positive
    df = pd.DataFrame({'Metric': [100, 80, 60, -50]}).rename(index=lambda x: str(x)).transpose()
    growth = calc_pos_neg_metrics(df.transpose(), 'Metric', years=3)
    # Expected: 10% (hardcoded rule)
    assert growth == 10

def test_calc_pos_neg_metrics_pos_to_neg():
    # Test change from positive to negative
    df = pd.DataFrame({'Metric': [-100, 80, 60, 50]}).rename(index=lambda x: str(x)).transpose()
    growth = calc_pos_neg_metrics(df.transpose(), 'Metric', years=3)
    # Expected: -10% (hardcoded rule)
    assert growth == -10
    
def test_calc_income_fcf_metrics(mock_income_df_good, mock_cashflow_df_good):
    rev_growth, eps_growth, fcf_growth = calc_income_fcf_metrics(mock_income_df_good, mock_cashflow_df_good, years=3)
    # Expected Revenue: (1000/500)^(1/3) - 1 ≈ 0.2599
    # Expected EPS: (20/5)^(1/3) - 1 ≈ 0.5874
    # Expected FCF: (60/10)^(1/3) - 1 ≈ 0.8171
    assert np.isclose(rev_growth, 0.2599, rtol=1e-3)
    assert np.isclose(eps_growth, 0.5874, rtol=1e-3)
    assert np.isclose(fcf_growth, 0.8171, rtol=1e-3)

def test_calc_margin_measures(mock_income_df_good):
    gross_margin_avg, net_margin_avg = calc_margin_measures(mock_income_df_good, years=3)
    # Gross Margins: 100/1000, 80/800, 60/600 -> all 0.1
    # Net Margins: 50/1000, 40/800, 30/600 -> all 0.05
    assert np.isclose(gross_margin_avg, 0.1)
    assert np.isclose(net_margin_avg, 0.05)

def test_calc_balance_measures(mock_balance_df_good):
    d_e_ratio, current_ratio, quick_ratio = calc_balance_measures(mock_balance_df_good)
    # D/E: -(700/500) = -1.4 (Debt/Equity from latest) -> Note: Total Debt (500) / Stockholders Equity (500) = 1.0; -1.0 expected
    # Total Debt is 500, Stockholders Equity is 500. D/E = 1.0. Function returns -1.0.
    # Current Ratio: 200/100 = 2.0
    # Quick Ratio: (200 - 50) / 100 = 1.5
    assert np.isclose(d_e_ratio, -1.0)
    assert np.isclose(current_ratio, 2.0)
    assert np.isclose(quick_ratio, 1.5)

def test_calc_altman_z(mock_income_df_good, mock_balance_df_good, mock_info_dict):
    # market_cap = 1,000,000
    # Total Assets = 1000
    # Total Liab = 500
    # EBIT = Gross Profit (100) - Assuming no other expenses for simplicity, but the mock DF only has 'EBIT' implicitly missing.
    # We will assume a hypothetical EBIT entry in mock_income_df_good for Z-score calculation.
    # For now, let's use the alternative in utils.py: Net Income (50) for 'C' calculation.

    # Modify mock income for EBIT presence check
    # income_with_ebit = mock_income_df_good.copy()
    # income_with_ebit['EBIT'] = income_with_ebit['Gross Profit'] # Simple proxy for test
    
    # Working Capital (Curr Assets - Curr Liab) = 200 - 100 = 100
    # A = 100 / 1000 = 0.1
    # B = Retained Earnings / Total Assets = 400 / 1000 = 0.4
    # C = EBIT / Total Assets = 100 / 1000 = 0.1
    # D = Market Cap / Total Liabilities = 1000000 / 500 = 2000
    # E = Total Revenue / Total Assets = 1000 / 1000 = 1.0
    # Z_score = 1.2*0.1 + 1.4*0.4 + 3.3*0.1 + 0.6*2000 + 1.0*1.0 = 0.12 + 0.56 + 0.33 + 1200 + 1.0 = 1202.01
    z_score = calc_altman_z(mock_income_df_good, mock_balance_df_good, mock_info_dict['marketCap'])
    assert np.isclose(z_score, 1202.01)

def test_calc_no_y_pos_msr(mock_income_df_good):
    # Basic EPS: [20, 15, 10, 5] -> all 4 are positive
    # Negative EPS: [10, -5, 20, 15] -> 3 positive
    df_neg = pd.DataFrame({'Basic EPS': [10, -5, 20, 15]}).transpose()
    assert calc_no_y_pos_msr(mock_income_df_good, 'Basic EPS') == 4
    assert calc_no_y_pos_msr(df_neg.transpose(), 'Basic EPS') == 3

def test_calc_shares_change(mock_income_df_good):
    # Shares: [100, 80, 60, 50]
    # Latest/Earliest - 1: (100/50) - 1 = 1.0
    # Expected: 1.0 (100% change)
    assert np.isclose(calc_shares_change(mock_income_df_good, 'Basic Average Shares'), 1.0)

def test_retrieve_info_data_quant(mock_info_dict):
    # Test when do_quant is True (input is a dictionary)
    marketCap, pb, ps, pe, ev_ebitda, beta = retrieve_info_data(mock_info_dict, True)
    assert marketCap == 1000000
    assert pb == -5.0
    assert pe == -20.0
    assert beta == -1.2
    assert ev_ebitda == -15.0

def test_retrieve_info_data_excel():
    # Test when do_quant is False (input is a DataFrame/Series read from excel)
    mock_info_excel = pd.DataFrame({
        'marketCap': [1000000],
        'priceToBook': [5.0],
        'priceToSalesTrailing12Months': [2.5],
        'forwardPE': [20.0],
        'beta': [1.2],
        'enterpriseToEbitda': [15.0]
    })
    marketCap, pb, ps, pe, ev_ebitda, beta = retrieve_info_data(mock_info_excel, False)
    assert marketCap == 1000000
    assert pb == -5.0
    assert pe == -20.0
    assert beta == -1.2
    assert ev_ebitda == -15.0

def test_retrieve_info_data_pe_fallback():
    # Test PE ratio fallback (forwardPE missing, trailingPE present)
    info = {'marketCap': 1, 'priceToBook': 1, 'priceToSalesTrailing12Months': 1, 'trailingPE': 10.0, 'beta': 1, 'enterpriseToEbitda': 1}
    marketCap, pb, ps, pe, ev_ebitda, beta = retrieve_info_data(info, True)
    assert pe == -10.0

# --- Unit Tests for Normalization ---

def test_normalize_dataframe_min_max():
    df = pd.DataFrame({'A': [10, 20, 30], 'B': [1, 2, 3]})
    normalized_df = normalize_dataframe(df.copy(), ['A', 'B'], [])
    # A: (10-10)/(30-10)=0, (20-10)/20=0.5, (30-10)/20=1.0
    # B: 0, 0.5, 1.0
    assert np.allclose(normalized_df['A'], [0.0, 0.5, 1.0])
    assert np.allclose(normalized_df['B'], [0.0, 0.5, 1.0])

def test_normalize_dataframe_z_score():
    df = pd.DataFrame({'A': [1, 2, 3]})
    normalized_df = normalize_dataframe(df.copy(), [], ['A'])
    # Mean = 2, Std = 1/sqrt(2) ≈ 0.707
    # (1-2)/0.707 ≈ -1.414, (2-2)/std = 0, (3-2)/0.707 ≈ 1.414
    expected_std = df['A'].std(ddof=1) # Pandas uses N-1 for std by default
    expected = (df['A'] - df['A'].mean()) / expected_std
    assert np.allclose(normalized_df['A'], expected)

def test_normalize_dataframe_constant_column():
    df = pd.DataFrame({'A': [10, 10, 10]})
    # Min-Max: Should result in 0 to avoid division by zero
    normalized_df_mm = normalize_dataframe(df.copy(), ['A'], [])
    assert np.allclose(normalized_df_mm['A'], [0, 0, 0])
    # Z-Score: Should result in 0 to avoid division by zero
    normalized_df_z = normalize_dataframe(df.copy(), [], ['A'])
    assert np.allclose(normalized_df_z['A'], [0, 0, 0])
    
if __name__ == "main":
    test_calc_pos_neg_metrics_positive_growth()
    test_calc_pos_neg_metrics_neg_to_pos()
    test_calc_pos_neg_metrics_pos_to_neg()
    # 1. Manually execute the fixture functions to get the DataFrames
    income_df = mock_income_df_good()
    balance_df = mock_balance_df_good()
    cashflow_df = mock_cashflow_df_good()
    info = mock_info_dict()
    # 2. Call the test function with the actual DataFrame objects
    rev_growth, eps_growth, fcf_growth = calc_income_fcf_metrics(income_df, cashflow_df, years=3)
    
    # 3. Now you can run the test assertion
    test_calc_income_fcf_metrics(income_df, cashflow_df)
    
    test_calc_margin_measures(income_df)
    test_calc_balance_measures(balance_df)
    test_calc_altman_z(income_df, balance_df, info)
    test_calc_no_y_pos_msr(income_df)
    test_calc_shares_change(income_df)
    test_retrieve_info_data_quant(info)
    test_retrieve_info_data_excel()
    test_retrieve_info_data_pe_fallback()
    test_normalize_dataframe_min_max()
    test_normalize_dataframe_z_score()
    test_normalize_dataframe_constant_column()
