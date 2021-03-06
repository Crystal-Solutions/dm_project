## Data Loading
pkgs <- c('tidyverse', 'corrplot', 'magrittr', 'zoo',  'RColorBrewer', 'gridExtra','MASS', 'randomForest')
invisible(lapply(pkgs, require, character.only = T))

setwd('J:/Raw/CS/Sem7/DM/project/data/')
#Train


preprocessData <- function(data_path, labels_path = NULL)
{
  # load data 
  df <- read.csv(data_path)
  
  # features we want
  features = c("reanalysis_specific_humidity_g_per_kg", 
               "reanalysis_dew_point_temp_k",
               "station_avg_temp_c", 
               "station_min_temp_c"
               )
  
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
preprocessData(data_path = 'dengue_features_train.csv', labels_path = 'part_removed/dengue_labels_train_filled_94_anom.csv') -> trains
sj_train <- trains[[1]]; iq_train <- as.data.frame(trains[2])


sj_train_subtrain <- head(sj_train, 800)
sj_train_subtest  <- tail(sj_train, nrow(sj_train) - 800)

iq_train_subtrain <- head(iq_train, 400)
iq_train_subtest  <- tail(iq_train, nrow(sj_train) - 400)


# function that returns Mean Absolute Error
mae <- function(error) return(mean(abs(error)) )

get_bst_model <- function(train, test, form)
{
  grid = 10 ^(seq(-8, -3,1))
  
  best_alpha = c()
  best_score = 1000
  
  # Step 2: Find the best hyper parameter, alpha
  for (i in grid)
  {
    model = glm.nb(formula = form,
                   data = train,
                   init.theta = i)
    
    results <-  predict(model, test)
    score   <-  mae(test$total_cases - results)
    
    if (score < best_score) {
      best_alpha <- i
      best_score <- score
      cat('\nbest score = ', best_score, '\twith alpha = ', best_alpha)
    }
  }
  
  # Step 3: refit on entire dataset
  combined <- rbind(train, test)
  combined_model = glm.nb(formula=form,
                          data = combined,
                          init.theta = best_alpha)
  
  return (combined_model)
}

getRandomForest <- function(train, test, form){
  r <- randomForest(Y~.,data=train, maxnodes=10)
  return (r)
}


# Step 1: specify the form of the model
form <- "total_cases ~ 1 +
  reanalysis_specific_humidity_g_per_kg +
  reanalysis_dew_point_temp_k + 
  station_avg_temp_c +
  station_min_temp_c 


sj_model <- get_bst_model(sj_train_subtrain, sj_train_subtest,form)
iq_model <- get_bst_model(iq_train_subtrain, iq_train_subtest,form)


# plot sj
sj_train$fitted = predict(sj_model, sj_train, type = 'response')
sj_train %>% 
  subset(year>1993) %>% 
  subset(weekofyear>1993) %>% 
  subset(year<1995) %>% 
  mutate(index = as.numeric(row.names(.))) %>%
  ggplot(aes(x = index)) + ggtitle("San Jose") +
  geom_line(aes(y = total_cases, colour = "total_cases")) + 
  geom_line(aes(y = fitted, colour = "fitted"))

# plot iq
iq_train$fitted = predict(iq_model, iq_train, type = 'response')
iq_train %>% 
  mutate(index = as.numeric(row.names(.))) %>%
  ggplot(aes(x = index)) + ggtitle("Iquitos") + 
  geom_line(aes(y = total_cases, colour = "total_cases")) + 
  geom_line(aes(y = fitted, colour = "fitted"))


# submitting the predictions
tests <- preprocessData('dengue_features_test.csv')
sj_test <- tests[[1]]; iq_test <- tests[[2]]

sj_test$predicted = predict(sj_model , sj_test, type = 'response')
iq_test$predicted = predict(iq_model , iq_test, type = 'response')

submissions = read.csv('submission_format.csv')
inner_join(submissions, rbind(sj_test,iq_test)) %>%
  dplyr::select(city, year, weekofyear, total_cases = predicted) ->
  predictions

predictions$total_cases %<>% round()
write.csv(predictions, 'submissions/predictions.csv', row.names = FALSE)



#Plot
sj_train %>% 
 # subset(year > 2001) %>% 
#  subset(year < 2003) %>% 
  subset(weekofyear >10) %>%
  mutate(index = as.numeric(row.names(.))) %>%
  ggplot(aes(x = index)) + ggtitle("San Jose") +
  geom_line(aes(y = total_cases, colour = "total_cases")) +
  geom_line(aes(y = fitted, colour = "fitted"))
#geom_line(aes(y = reanalysis_specific_humidity_g_per_kg, colour = "fitted")) 
# geom_line(aes(y = reanalysis_dew_point_temp_k, colour = "fitted"))+ 
# geom_line(aes(y = station_avg_temp_c, colour = "fitted"))+ 
# geom_line(aes(y = station_min_temp_c, colour = "fitted"))


# plot iq
iq_train$fitted = predict(iq_model, iq_train, type = 'response')
iq_train %>% 
  subset(year > 1990) %>% 
  subset(year < 2010) %>% 
  mutate(index = as.numeric(row.names(.))) %>%
  ggplot(aes(x = index)) + ggtitle("Iquitos") + 
  geom_line(aes(y = total_cases, colour = "total_cases")) + 
  geom_line(aes(y = fitted, colour = "fitted"))


