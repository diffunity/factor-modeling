import yaml
import logging.config
from pipeline import Data_pipeline

def main():

  configs = yaml.load(open("./settings.yml").read(), Loader=yaml.Loader)
  
  data_config = configs["data_config"]
  logging_config = configs["logging_config"]
  
  logging.config.dictConfig(logging_config)
  logger = logging.getLogger('root')
  
  data = Data_pipeline(data_config, logger)

  data.plot(quote=data_config["plot_stock_graph"]["quote"], \
            feature=data_config["plot_stock_graph"]["feature"], \
            ma=data_config["plot_stock_graph"]["ma"], \
            filename=data_config["plot_stock_graph"]["filename"])

  # volatilities
  vol_data = data.fund_to_vol()
  raw_vol_data = vol_data.raw()
  minmax_vol_data = vol_data.minmax()
  std_vol_data = vol_data.standardscaling()
  
  # percentage change
  pct_data = data.fund_to_pct_change(data_config["report"]["pct_change"])
  raw_pct_data = pct_data.raw()
  minmax_pct_data = pct_data.minmax()
  std_pct_data = pct_data.standardscaling()

  data.produce_report(raw_vol_data.dropna(axis=1), 
                      test_ratio=data_config["report"]["test_ratio"],
                      topn=data_config["report"]["topn"],
                      fname="Raw_Vol.jpg")
  data.produce_report(minmax_vol_data, 
                      test_ratio=data_config["report"]["test_ratio"],
                      topn=data_config["report"]["topn"],
                      fname="MinMax_Vol.jpg")
  data.produce_report(std_vol_data, 
                      test_ratio=data_config["report"]["test_ratio"],
                      topn=data_config["report"]["topn"],
                      fname="StandardScaled_Vol.jpg")

  data.produce_report(raw_pct_data.dropna(axis=1), 
                      test_ratio=data_config["report"]["test_ratio"],
                      topn=data_config["report"]["topn"],
                      fname="Raw_Pct_Change.jpg")
  data.produce_report(minmax_pct_data,
                      test_ratio=data_config["report"]["test_ratio"],
                      topn=data_config["report"]["topn"],
                      fname="MinMax_Pct_Change.jpg")
  data.produce_report(std_pct_data, 
                      test_ratio=data_config["report"]["test_ratio"],
                      topn=data_config["report"]["topn"],
                      fname="StandardScaled_Pct_Change.jpg")
  logger.info("Finished Producing reports")

if __name__=="__main__":
  main()
