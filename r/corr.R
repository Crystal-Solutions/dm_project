## Data Loading
pkgs <- c('tidyverse', 'corrplot', 'magrittr', 'zoo',  'RColorBrewer', 'gridExtra','MASS')
invisible(lapply(pkgs, require, character.only = T))

setwd('J:/Raw/CS/Sem7/DM/project/data/original_data/')
#Train


preprocessData <- function(data_path, labels_path = NULL)
{
  # load data 
  df <- read.csv(data_path)
  
  # features we want
  features = c("reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k",
               "station_avg_temp_c", "station_min_temp_c", "reanalysis_tdtr_k")
  
  # fill missing values
  df[features] %<>% na.locf(fromLast = TRUE) 
  
  # add city if labels data aren't provided
  if (is.null(labels_path)) features %<>% c("city", "year", "weekofyear")
  
  # select features we want
  df <- df[features]
  
  # add labels to dataframe
  if (!is.null(labels_path)) df %<>% cbind(read.csv(labels_path))
  
  # filter by city
  df_sj <- filter(df, city == 'sj')
  df_iq <- filter(df, city == 'iq')
  
  # return a list with the 2 data frames 
  return(list(df_sj, df_iq))
}

# preprocess the .csv files
preprocessData(data_path = 'dengue_features_train.csv', labels_path = 'dengue_labels_train.csv') -> trains
sj_train <- trains[[1]]; iq_train <- as.data.frame(trains[2])

cc_result <- acf(sj_train,5,plot=TRUE)

