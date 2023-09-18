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
  select(-'registered', -'casual')

# Feature Engineering
my_recipe <- recipe(count~., data=bike_cleaned) %>% # Set model formula and data
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # where weather is 4 change to 3 (only one value like that)
  step_date(datetime, features="dow") %>% # gets day of week
  step_time(datetime, features="hour") %>% # gets hour
  #step_dummy(all_nominal_predictors()) #create dummy variables
  step_rm('workingday')
prepped_recipe <- prep(my_recipe) # Sets up the pre-processing
bake(prepped_recipe, new_data=bike_cleaned)


### Linear Regression ###
my_mod <- linear_reg() %>% #Type of model
  set_engine("lm") # Engine = What R function to use

#Set up the workflow
bike_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_mod) %>%
  fit(data = bike_cleaned) # Fit the workflow to training data

#### show the model
extract_fit_engine(bike_workflow) %>%
  summary()

bike_predictions_l <- predict(bike_workflow,
                            new_data=bike_test) # Use fit to predict on test data

predictions <- bike_predictions_l %>% 
  mutate(datetime = bike_test$datetime) %>%
  mutate(count = ifelse(.pred < 0, 0, .pred)) %>% # round all negative predictions to 0
  select(datetime, count)
predictions$datetime <- as.character(format(predictions$datetime))

vroom_write(predictions, 'BikeSharePreds.csv', delim = ",")

### Poisson Regression ###

library(poissonreg)

pois_mod <- poisson_reg() %>% #Type of model
  set_engine("glm") # GLM = generalized linear model

# Set up workflow
bike_pois_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(pois_mod) %>%
  fit(data = bike_cleaned) # Fit the workflow

bike_predictions_p <- predict(bike_pois_workflow,
                            new_data=bike_test) # Use fit to predict12

predictions <- bike_predictions_p %>% 
  mutate(datetime = bike_test$datetime) %>%
  mutate(count = ifelse(.pred < 0, 0, .pred)) %>% # round all negative predictions to 0
  select(datetime, count)
predictions$datetime <- as.character(format(predictions$datetime))

vroom_write(predictions, 'BikeSharePreds.csv', delim = ",")
