import yaml
import logging.config
from pipeline import data_pipeline

def main():

  configs = yaml.load(open("./settings.yml").read(), Loader=yaml.Loader)
  
  data_config = configs["data_config"]
  logging_config = configs["logging_config"]
  
  logging.config.dictConfig(logging_config)
  logger = logging.getLogger('root')
  
  data = data_pipeline(data_config, logger)

  logger.info(data.companies)
  
  logger.info(data.stock_volatilities([5,15,30,60]).head(5))

  logger.info(data.OLS(['Capital Expenditures',\
                        'Investments',\
                        'Other Investing Activities',\
                        'Net Cash Flows-Investing',\
                        'Effect of Exchange Rate',\
                        'Other Equity',\
                        'Treasury Stock',\
                        'Sale and Purchase of Stock',\
                        'Gross Margin',\
                        'Operating Margin',\
                        'Misc. Stocks',\
                        'Quick Ratio',\
                        'Current Ratio',\
                        'Cash Ratio']).summary())


if __name__=="__main__":
  main()
