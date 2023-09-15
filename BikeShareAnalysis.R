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
  mutate(weather = ifelse(weather == 4, 3, weather)) %>% # where weather is 4 change to 3 (only one value like that)
  select(-'registered', -'casual')
bike_test_cleaned <- bike_test %>%
  mutate(weather = ifelse(weather == 4, 3, weather))



# Feature Engineering
my_recipe <- recipe(count~., data=bike_cleaned) %>% # Set model formula and data
  step_date(datetime, features="dow") %>% # gets day of week
  step_time(datetime, features="hour") %>% # gets hour
  step_dummy(all_nominal_predictors()) #create dummy variables
prepped_recipe <- prep(my_recipe) # Sets up the pre-processing
bake(prepped_recipe, new_data=bike_cleaned)

# Linear Regression
my_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine = What R function to use

bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike_cleaned) # Fit the workflow to training data

bike_predictions <- predict(bike_workflow,
                            new_data=bike_test_cleaned) # Use fit to predict on test data

predictions <- bike_predictions %>% 
  mutate(datetime = bike_test_cleaned$datetime) %>%
  mutate(count = ifelse(.pred < 0, 0, .pred)) %>% # round all negative predictions to 0
  select(datetime, count)
predictions$datetime <- as.character(format(predictions$datetime))

vroom_write(predictions, 'BikeSharePreds.csv', delim = ",")


