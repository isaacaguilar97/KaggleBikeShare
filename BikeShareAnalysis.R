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
  select(-'registered', -'casual') %>%
  mutate(lg_count = log(count)) %>% #Create the log variable of count
  select(-count)

# Feature Engineering
my_recipe <- recipe(lg_count~., data=bike_cleaned) %>% # Set model formula and data
  step_mutate(weather = ifelse(weather == 4, 3, weather)) %>% # where weather is 4 change to 3 (only one value like that)
  step_date(datetime, features="dow") %>% # gets day of week
  step_time(datetime, features="hour") %>% # gets hour
  step_dummy(all_nominal_predictors()) %>% #create dummy variables
  step_normalize(all_numeric_predictors()) %>% # Make mean 0, sd=1
  step_rm('workingday', 'datetime')
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

### Penalized Regression ###
library(tidymodels)
library(poissonreg) #if you want to do penalized, poisson regression

### Penalized regression model ###
preg_model <- linear_reg(penalty=0, mixture=0) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model) %>%
  fit(data=bike_cleaned)
bike_pred_penilized <- predict(preg_wf, new_data=bike_test)

# clean format
predictions <- bike_pred_penilized %>% 
  mutate(datetime = bike_test$datetime) %>%
  mutate(count = exp(.pred)) %>% # exponentiate lg_count
  select(datetime, count)
predictions$datetime <- as.character(format(predictions$datetime))

vroom_write(predictions, 'BikeSharePreds.csv', delim = ",")

### Tuning Parameters ###
library(tidymodels)
library(poissonreg) #if you want to do penalized, poisson regression

## Penalized regression model
preg_model <- linear_reg(penalty=tune(),
                         mixture=tune()) %>% #Set model and tuning
  set_engine("glmnet") # Function to fit in R

## Set Workflow
preg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(preg_model)

## Grid of values to tune over14
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(bike_cleaned, v = 5, repeats=1)

## Run the CV
CV_results <- preg_wf %>%
  tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(rmse, mae)) #Or leave metrics NULL

## Plot Results (example)
collect_metrics(CV_results) %>% # Gathers metrics into DF
  filter(.metric=="rmse") %>%
  ggplot(data=., aes(x=penalty, y=mean, color=factor(mixture))) +
  geom_line()

## Find Best Tuning Parameters
bestTune <- CV_results %>% 
  select_best("rmse")

## Finalize the Workflow & fit it
final_wf <- preg_wf %>% 
  finalize_workflow(bestTune) %>% 
  fit(data=bike_cleaned)

## Predict
bike_pred_tunned <- final_wf %>%
  predict(new_data = bike_test)

# clean format
predictions <- bike_pred_tunned %>% 
  mutate(datetime = bike_test$datetime) %>%
  mutate(count = exp(.pred)) %>% # exponentiate lg_count
  select(datetime, count)
predictions$datetime <- as.character(format(predictions$datetime))

vroom_write(predictions, 'BikeSharePreds.csv', delim = ",")
