### Model Documentation (Approach)

The framework employs a three-stage approach to rank stocks, combining quantitative, qualitative, and external factors. The final rank is a composite average of the individual ranks derived from each of these categories.

#### 1. Quantitative Analysis
This part of the model assesses a company's financial health based on its fundamental data. It retrieves income statements, balance sheets, and cash flow statements from Yahoo Finance. The model calculates various metrics such as revenue growth, earnings per share (EPS) growth, free cash flow (FCF) growth, average margin or Altman Z score. Each of these metrics is then ranked across the selected group of stocks.

The quantitative score is a **weighted average** of the individual metrics, where the weights are defined by the user in the `quant_weights` dictionary in `params.yaml`.
Finally, the rank of each stock is determined, which further enters Final rank estimation.

The metrics, which are taken into account:
- revenue growth, eps growth, fcf growth - average across available period in percentage points. In case the metrics turns positive from negative, it is awarded with 10%, otherwise -10%
- gross / net margin - average across available period in percentage points
- D / E, current ratio, quick ratio - balance sheet measures multiplied by 10. D / E enters with negative sign given its reverse impact on the company
- Altman Z score
- number of years with positive EPS / FCF - both enter the score multiplied by 10
- % change in shares outstanding - multiplied by 100 (in percentage points), enters with negative sign given its reverse impact on the stock price
- valuation metrics (P/E, P/B, P/S, EV/EBITDA)
- volatiltiy metrics (beta, EPS st deviation to mean ratio)
- accruals Ratio (earnings vs. cash flow)
- cash conversion ratio (CFO / Net Income)

#### 2. Qualitative Analysis
The qualitative analysis focuses on non-numerical factors that can provide a competitive advantage, often referred to as a "moat." This part of the model performs sentiment analysis on earnings call transcripts. A qualitative score is created by combining scores from sentiment, moat, and leadership factors. The qualitative rank is then calculated based on this combined score.

The metrics, which are taken into account:
- MOAT Quant score - points awarded for sufficiently high gross margin and ROIC;
- MOAT Text Score - points awarded for matching the previously defined words defining MOAT within earning call transcript;
- Leadership Text Score - points awarded for matching the previously defined words defining strong management within earning call transcript;
- Sentiment Score - additional points awarded for a difference between positive and negative sentyment within earnings call transcript measure by nltk packages Sentiment Analyzer;

#### 3. External Ratings
The framework integrates external expertise by parsing a provided PDF document. The `utils_parse_Cymcyk.py` script is specifically designed to extract analyst ratings from the PDF file. These ratings are then assigned to their respective tickers and used to generate a separate "External Rating Score." If a ticker is not found in the PDF, it is assigned the lowest possible rating (-2).

#### Final Ranking
The final rank for each stock is an average of the ranks from the three components: Quantitative Rank, Qualitative Rank, and External Rating Rank. If external ratings are not used, the final rank is an average of only the Quantitative and Qualitative ranks. This approach provides a holistic, multi-faceted perspective for stock selection.