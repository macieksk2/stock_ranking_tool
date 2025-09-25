# Stock Ranking Framework
This project is a Python-based framework for calculating the average rank of selected stocks. The ranking is based on a comprehensive model that integrates three key factors: quantitative financial data, qualitative analysis from earnings call transcripts, and external ratings from a PDF source prepared by a stock analyst.

### Features
* **Quantitative Analysis**: Fetches financial data (income, balance, cash flow) for multiple tickers from Yahoo Finance and ranks them based on user-defined weights.
* **Qualitative Analysis**: Analyzes earnings call transcripts for sentiment and qualitative "moat" factors.
* **External Ratings**: Parses a PDF document to extract and integrate external analyst ratings into the final score.
* **Customizable**: All tickers, scoring weights, and data sources are easily configurable in the `params.yaml` file.

### Prerequisites
Before running the script, ensure you have Python 3.x installed. The required libraries can be installed using the `requirements.txt` file.

### Installation
1.  Clone the repository or download the project files.
2.  Install the dependencies by running the following command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```
3.  Place the `StockScan_grudzien_2024_fin-laqvwb.pdf` file in the correct location for the script to access it.

### Configuration
All user-specific inputs are managed in `params.py`. You can modify the following:
* `tickers`: A list of stock tickers you want to analyze.
* `DO_PULL_QUANT`: Set to `True` to pull quantitative data from Yahoo Finance; set to `False` to use local files.
* `DO_PULL_QUAL` : Set to `True` to pull qualitative data from online sources; set to `False` to use local files.
* `DO_EXT_RATING`: Set to `True` to parse the external ratings from the PDF file.
#### QUANTITATIVE PART
* `quant_weights`: A dictionary to adjust the weighting of each quantitative financial metric (e.g., `revenue_growth`, `eps_growth`). The weights should sum to 1.0.
#### QUALITATIVE PART
* `quant_moat_params`     : Set a minimum gross margin and ROIC to be qualified as a company with a MOAT
* `quant_moat_scores`     : Set scores assigned to company for breaching minimum gross margin and ROIC level
* `moat_keywords`         : Define words indicating a strength of MOAT to be found in earnings call transcript
* `leadership_keywords`   : Define words indicating a strength of leadership of the company's Board of Directors to be found in earnings call transcript
* `keyword_score_weight`  : Set a weight of number of leadership keywords when assessing management strength
* `sentiment_score_weight`: Set a weight of sentiment measured with nltk package SentimentIntensityAnalyzer function
* `qual_scorecard_weights`: A dictionary to adjust the weighting of each qualitative financial metric (MOAT from transcript, MOAT in quantitative data, management strength from transcript and sentiment). The weights should sum to 1.0.
* `inv_websites`          : A dctionary mapping stock tickers to URLs with latest earning call transcript (currently based use Motley Fool as a source)

#### EXTERNAL RATING PART
* `ticker_mapping`: A dictionary to map tickers to the company names used in the external PDF, ensuring the parser finds the correct ratings.
* `pdf_path`: A path to folder storing pdf with external ratings of stocks

### Usage
To run the analysis, execute the `main.py` script from your terminal:
```bash
python main.py

The script will perform the analysis based on the settings in params.py and print the final ranked summary of the selected stocks to the console and store it in csv in output folder. 
It will also generate output files in the output/ directory.

Project Structure
.
├── main.py                    # Main script to run the analysis
└── requirements.txt           # List of required Python libraries
input
├── params.yaml                # User input and configuration
├── StockScan_grudzien_2024_fin-laqvwb.pdf # Input file with external ratings
utils
├── utils.py                   # Helper functions for quantitative, qualitative analysis or parsing external rating
output
├── figures                    # Folder storing intermediate output in the form of plots (quant part, qual part, external rating)
├── stock_qual_scores          # Folder storing intermediate output of qualitative analysis
├── stock_quant_scores         # Folder storing intermediate output of quantitative analysis
├── stock_analysis_scores.csv  # Final output including a dataframe with stocks and their scores and ranks per category
├── Stock_Performance_Report.pdf  # Pdf report including the most relevant plots from folder figures
├── Stock_Quant_attribution.pdf   # Pdf report including the watterfal plots with attribution of metrics contributing to final Quant score per stock

