##
## BikeShareAnalysis
##

## Loading Libraries
library(tidyverse)
library(vroom)
library(tidymodels)

# Load data
bike <- vroom('./train.csv')
bike_test <- vroom('./test.csv')

# Clean data
bike_cleaned <- bike %>%
  mutate(weather = ifelse(weather == 4, 3, weather))# where weather is 4 change to 3 (only one value like that)




# Feauter Engineering
my_recipe <- recipe(count~., data=bike_cleaned) %>% # Set model formula and data
  step_date(datetime, features="dow") %>% # gets day of week
  step_time(datetime, features="hour") %>% # gets hour
  step_dummy(all_nominal_predictors()) %>% #create dummy variables
  step_select(everything(), -'registered', -'casual') #selects columns
prepped_recipe <- prep(my_recipe) # Sets up the preprocessing using myData
bake(prepped_recipe, new_data=bike)



