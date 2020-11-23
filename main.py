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

  # volatilities
  vol_data = data.fund_to_vol()
  raw_vol_data = vol_data.raw()
  minmax_vol_data = vol_data.minmax()
  std_vol_data = vol_data.standardscaling()
  
  # percentage change
  pct_data = data.fund_to_pct_change(5)
  raw_pct_data = pct_data.raw()
  minmax_pct_data = pct_data.minmax()
  std_pct_data = pct_data.standardscaling()


  data.produce_report("Raw_Vol.jpg", raw_vol_data)
  data.produce_report("MinMax_Vol.jpg", minmax_vol_data)
  data.produce_report("StandardScaled_Vol.jpg", std_vol_data)

  data.produce_report("Raw_Pct_Change.jpg", raw_pct_data)
  data.produce_report("MinMax_Pct_Change.jpg", minmax_pct_data)
  data.produce_report("StandardScaled_Pct_Change.jpg", std_pct_data)

if __name__=="__main__":
  main()
