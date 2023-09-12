##
## BikeShareEDA
##

## Loading Libraries
library(tidyverse)
library(vroom)
library(DataExplorer)
library(patchwork)
?patchwork

# Load data
bike <- vroom('./train.csv')

## EDA

dplyr::glimpse(bike) # lists the variable type of each column
skimr::skim(bike)
plot1 <- plot_intro(bike) # visualization of glimpse()
plot3 <- plot_correlation(bike) # correlation heat map between variables
plot2 <- plot_bar(bike) # bar charts of all discrete variables
plot_histogram(bike) # histograms of all numerical variables
<- plot_missing(bike) # percent missing in each column
GGally::ggpairs(bike) # 1/2 scatterplot and 1/2 correlation heat map # Takes a lot of time to load

# Relationship between windspeed and Count
plot4 <- ggplot(data=bike, aes(x=windspeed, y=count)) +
  geom_point() +
  geom_smooth(se=FALSE)

# 4 Panel ggplot
(plot1 + plot2) / (plot3 + plot4)


