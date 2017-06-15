## Data Loading
pkgs <- c('tidyverse', 'corrplot', 'magrittr', 'zoo',  'RColorBrewer', 'gridExtra','MASS')
invisible(lapply(pkgs, require, character.only = T))

setwd('J:/Raw/CS/Sem7/DM/project/data/original_data/')

## Data Loading
train_features = read.csv('dengue_features_train.csv')
train_labels   = read.csv('dengue_labels_train.csv')

head(train_features[1:8])

# Seperate data by city
sj_train_features = train_features %>% filter(city == 'sj')
sj_train_labels   = train_labels   %>% filter(city == 'sj')

iq_train_features = train_features %>% filter(city == 'iq')
iq_train_labels   = train_labels   %>% filter(city == 'iq')


# data shape for each city
cat('\nSan Juan\n',
    '\t features: ', sj_train_features %>% ncol, 
    '\t entries: ' , sj_train_features %>% nrow,
    '\t labels: '  , sj_train_labels %>% nrow)

cat('\nIquitos\n',
    '\t features: ', iq_train_features %>% ncol, 
    '\t entries: ' , iq_train_features %>% nrow,
    '\t labels: '  , iq_train_labels %>% nrow)

sj_train_features %<>% dplyr::select(-week_start_date)
iq_train_features %<>% dplyr::select(-week_start_date)
# 
# apply(sj_train_features, 2, function(x) 
#   round(100 * (length(which(is.na(x))))/length(x) , digits = 1)) %>%
#   as.data.frame() %>%
#   `names<-`('Percent of Missing Values')

allFIelds <- colnames(sj_train_features)

# impute NAs by the latest value
sj_train_features[allFIelds] %<>% na.locf(fromLast = TRUE)
iq_train_features[allFIelds] %<>% na.locf(fromLast = TRUE)

sj_train_features %>%
  mutate(index = as.numeric(row.names(.))) %>%
  ggplot(aes(index, ndvi_ne)) + 
  geom_line(colour = 'dodgerblue') +
  ggtitle("Vegetation Index over Time")


# total cases of dengue: histograms
rbind(iq_train_labels, sj_train_labels) %>% 
  ggplot(aes(x = total_cases,fill = ..count..)) + 
  geom_histogram(bins = 12, colour = 'black') + ggtitle('Total Cases of Dengue') +
  scale_y_continuous(breaks = seq(0,700,100)) + facet_wrap(~city)


# corerlations between features
sj_train_features %<>% mutate('total_cases' = sj_train_labels$total_cases)
iq_train_features %<>% mutate('total_cases' = iq_train_labels$total_cases)

# plot san juan correlation matrix
sj_train_features %>% 
  dplyr::select(-city, -year, -weekofyear) %>%
  cor(use = 'pairwise.complete.obs') -> M1

corrplot(M1, type="lower", method="color",
         col=brewer.pal(n=8, name="RdBu"),diag=FALSE)
# plot iquitos correlation matrix
iq_train_features %>% 
  dplyr::select(-city, -year, -weekofyear) %>%
  cor(use = 'pairwise.complete.obs') -> M2

corrplot(M2, type="lower", method="color",
         col=brewer.pal(n=8, name="RdBu"),diag=FALSE)


# see the correlations as barplot
sort(M1[21,-21]) %>%  
  as.data.frame %>% 
  `names<-`('correlation') %>%
  ggplot(aes(x = reorder(row.names(.), -correlation), y = correlation, fill = correlation)) + 
  geom_bar(stat='identity', colour = 'black') + scale_fill_continuous(guide = FALSE) + scale_y_continuous(limits =  c(-.15,.25)) +
  labs(title = 'San Jose\n Correlations', x = NULL, y = NULL) + coord_flip() -> cor1

# can use ncol(M1) instead of 21 to generalize the code
sort(M2[21,-21]) %>%  
  as.data.frame %>% 
  `names<-`('correlation') %>%
  ggplot(aes(x = reorder(row.names(.), -correlation), y = correlation, fill = correlation)) + 
  geom_bar(stat='identity', colour = 'black') + scale_fill_continuous(guide = FALSE) + scale_y_continuous(limits =  c(-.15,.25)) +
  labs(title = 'Iquitos\n Correlations', x = NULL, y = NULL) + coord_flip() -> cor2

grid.arrange(cor1, cor2, nrow = 1)


ccf(sj_train_features$reanalysis_specific_humidity_g_per_kg,sj_train_labels$total_cases)




