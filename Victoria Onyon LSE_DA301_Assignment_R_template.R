## LSE Data Analytics Online Career Accelerator 

# DA301:  Advanced Analytics for Organisational Impact

###############################################################################

# Assignment template

## Scenario
## You are a data analyst working for Turtle Games, a game manufacturer and 
## retailer. They manufacture and sell their own products, along with sourcing
## and selling products manufactured by other companies. Their product range 
## includes books, board games, video games and toys. They have a global 
## customer base and have a business objective of improving overall sales 
##performance by utilising customer trends. 

## In particular, Turtle Games wants to understand:
## - how customers accumulate loyalty points (Week 1)
## - how useful are remuneration and spending scores data (Week 2)
## - can social data (e.g. customer reviews) be used in marketing 
##     campaigns (Week 3)
## - what is the impact on sales per product (Week 4)
## - the reliability of the data (e.g. normal distribution, Skewness, Kurtosis)
##     (Week 5)
## - if there is any possible relationship(s) in sales between North America,
##     Europe, and global sales (Week 6).

################################################################################

# Week 4 assignment: EDA using R

## The sales department of Turtle games prefers R to Python. As you can perform
## data analysis in R, you will explore and prepare the data set for analysis by
## utilising basic statistics and plots. Note that you will use this data set 
## in future modules as well and it is, therefore, strongly encouraged to first
## clean the data as per provided guidelines and then save a copy of the clean 
## data for future use.

# Instructions
# 1. Load and explore the data.
##  - Remove redundant columns (Ranking, Year, Genre, Publisher) by creating 
##      a subset of the data frame.
##  - Create a summary of the new data frame.
# 2. Create plots to review and determine insights into data set.
##  - Create scatterplots, histograms and boxplots to gain insights into
##      the Sales data.
##  - Note your observations and diagrams that could be used to provide
##      insights to the business.
# 3. Include your insights and observations.

###############################################################################

# Install and import Tidyverse.
library('tidyverse')

# Import the data set.
sales <- read.csv(file.choose(), header=T)

# Print the data frame.
print(sales)
head(sales)
as_tibble(sales)
View(sales)

# Create a new data frame from a subset of the sales data frame.
# Remove unnecessary columns. 
sales_clean <- subset(sales, select = -c(Ranking, Year, Genre, Publisher))

# Convert product column type to string as Product ID is not a numeric value
# But the ID of the product 
sales_clean$Product = as.character((sales_clean$Product)) 

# View the data frame.
View(sales_clean)
head(sales_clean)

# View the descriptive statistics.
summary(sales_clean)
dim(sales_clean)

# Save the data set as a csv file.
write.csv(sales_clean, 'sales_clean.csv')

################################################################################

# 2. Review plots to determine insights into the data set.

## 2a) Scatterplots
# Create scatterplots.
# Comparing NA and EU sales
qplot(NA_Sales, EU_Sales, data=sales_clean, color=Platform)

# Investigating global sales by platform
qplot(Platform, Global_Sales, 
      data=sales_clean, 
      geom=c('point', 'jitter'))

## 2b) Histograms
# Create histograms.
# Analysing the distribution of sales by region
qplot(Global_Sales, bins=20, data=sales_clean)
qplot(NA_Sales, bins=20, data=sales_clean)
qplot(EU_Sales, bins=20, data=sales_clean)

## 2c) Boxplots
# Create boxplots.
# Analysing sales by region
qplot(Global_Sales,data=sales_clean, geom='boxplot')
qplot(NA_Sales,data=sales_clean, geom='boxplot')
qplot(EU_Sales,data=sales_clean, geom='boxplot')


# 2d. Bar chart to invsestigate the number of products per platform offered 
# by Turtle Gams
qplot(Platform, data=sales_clean, geom='bar')

###############################################################################

# 3. Observations and insights

## Your observations and insights here ......
# The scatterplot shows that the majority of products have similar sales in the 
# EU and North America. The scatterplot also shows how the sales for one Wii 
# product far exceed any other products and there are a few products which have
# very high North_America sales but low European sales. 
# Interestingly the platforms which have the most products offered by Turtle 
# Games are not the platforms of the top selling products. The platforms which
# have the most products offered by Turtle Games are X360, PS3 and PS4. 
# Whereas the best selling products are on Wii, SNES and NES. 



###############################################################################
###############################################################################


# Week 5 assignment: Cleaning and maniulating data using R

## Utilising R, you will explore, prepare and explain the normality of the data
## set based on plots, Skewness, Kurtosis, and a Shapiro-Wilk test. Note that
## you will use this data set in future modules as well and it is, therefore, 
## strongly encouraged to first clean the data as per provided guidelines and 
## then save a copy of the clean data for future use.

## Instructions
# 1. Load and explore the data.
##  - Continue to use the data frame that you prepared in the Week 4 assignment. 
##  - View the data frame to sense-check the data set.
##  - Determine the `min`, `max` and `mean` values of all the sales data.
##  - Create a summary of the data frame.
# 2. Determine the impact on sales per product_id.
##  - Use the group_by and aggregate functions to sum the values grouped by
##      product.
##  - Create a summary of the new data frame.
# 3. Create plots to review and determine insights into the data set.
##  - Create scatterplots, histograms, and boxplots to gain insights into 
##     the Sales data.
##  - Note your observations and diagrams that could be used to provide 
##     insights to the business.
# 4. Determine the normality of the data set.
##  - Create and explore Q-Q plots for all sales data.
##  - Perform a Shapiro-Wilk test on all the sales data.
##  - Determine the Skewness and Kurtosis of all the sales data.
##  - Determine if there is any correlation between the sales data columns.
# 5. Create plots to gain insights into the sales data.
##  - Compare all the sales data (columns) for any correlation(s).
##  - Add a trend line to the plots for ease of interpretation.
# 6. Include your insights and observations.

################################################################################

# 1. Load and explore the data

# View data frame created in Week 4.
as_tibble(sales_clean)
str(sales_clean)

# Check output: Determine the min, max, and mean values.
min(sales_clean$NA_Sales)
min(sales_clean$EU_Sales)
min(sales_clean$Global_Sales)
max(sales_clean$NA_Sales)
max(sales_clean$EU_Sales)
max(sales_clean$Global_Sales)
mean(sales_clean$Global_Sales)
mean(sales_clean$NA_Sales)
mean(sales_clean$EU_Sales)
  

# View the descriptive statistics.
summary(sales_clean)
DataExplorer::create_report(sales_clean)

###############################################################################

# 2. Determine the impact on sales per product_id.

## 2a) Use the group_by and aggregate functions.

# Group data based by Platform and determine the sum per Product.
platform_sales <- sales_clean %>% group_by(Platform) %>%
  summarise(total_sales = sum(Global_Sales),
            NA_sales = sum(NA_Sales),
            EU_sales = sum(EU_Sales),
            .groups='drop')


# Explore the data frame.
View(platform_sales)
summary(platform_sales)

# Using the aggregate method to group data by sales and 
product_sales <- aggregate(cbind(NA_Sales, EU_Sales, Global_Sales)~Product,
                           sales_clean, sum)
View(product_sales)

## 2b) Determine which plot is the best to compare game sales.
# Create scatterplots.
platform_sales_plot <- ggplot(platform_sales,mapping=aes(x= NA_sales, 
                                                         y=EU_sales), 
                      fill(Platform)) +
         geom_point(alpha=0.75,
                    size=2.5) +
  # Add a title and subtitle.
         labs(title=" Sales by Platform in North America vs Europe",
              x = 'North America Sales',
              y = 'European Sales') +
  geom_point() 

# Print the plot 
platform_sales_plot

# Specify the ggplotly() function and pass the plot to make interactive
ggplotly(platform_sales_plot) 


# Create histograms.
platform_products <- ggplot(sales_clean, aes(x=Platform)) +
  geom_histogram( fill ='orange',
    stat='count') +
  # Add a title and subtitle.
labs(title=" Number of Products by Platform",
     x = 'Platform',
     y = 'Number of Products')

# Create boxplots


###############################################################################


# 3. Determine the normality of the data set.

## 3a) Create Q-Q Plots
# Create Q-Q Plots.
qqnorm(sales_clean$Global_Sales)
qqline(sales_clean$Global_Sales) 

## 3b) Perform Shapiro-Wilk test
# Install and import Moments.
library(moments)

# Perform Shapiro-Wilk test.
shapiro.test(sales_clean$Global_Sales)

## 3c) Determine Skewness and Kurtosis
# Skewness and Kurtosis.
# Install the moments package and load the library.
skewness(sales_clean$Global_Sales) 
kurtosis(sales_clean$Global_Sales)

## 3d) Determine correlation
# Determine correlation be
# Create a subset of the data for just sales data
only_sales <- subset(sales_clean,select=c(3,4,5))
head(only_sales)
cor(only_sales)

# Visualise the correlation using corPlot()
# Import the pysch package
library(psych)

# Use the corPlot() function and set character size
corPlot(only_sales, cex=1)


###############################################################################

# 4. Plot the data
# Create plots to gain insights into data.
# Choose the type of plot you think best suits the data set and what you want 
# to investigate. Explain your answer in your report.

# Load the plotly and reshape library
library(plotly)
library(reshape2)

# Plotting product sales by region 
sales_by_region <- ggplot(sales_clean, aes(x = NA_Sales, y = EU_Sales,)) +
  geom_point(size=3, 
             alpha = 0.75,
             color='blue') +
  geom_smooth(method = lm)+
  labs(title="North America vs EU Sales of Products",
       x = "North America Sales in £",
       y = "EU Sales in £") +
  theme_classic()

# Print the plot 
sales_by_region

# Specify the ggplotly() function and pass the plot to make interactive
ggplotly(sales_by_region) 

# Plotting the top selling products by EU and NA
# Using arrange() and slice to get the top global selling products
top_GS <- sales_clean %>%
  arrange(desc(Global_Sales)) %>%
  slice(1:10) 
View(top_GS)

# Plotting the top selling products by region 
top_sales_by_region <- ggplot(top_GS, aes(x = NA_Sales, y = EU_Sales,
                                          fill=Platform)) +
  geom_point(size=5, 
             alpha = 0.75) +
  labs(title="Global Top Selling Products - North America vs EU Sales",
       x = "North America Sales in £",
       y = "EU Sales in £") +
  theme_classic()

# Print the plot 
top_sales_by_region
   
# Specify the ggplotly() function and pass the plot to make interactive
ggplotly(top_sales_by_region) 


top_sales_by_region <- ggplot(top_GS, aes(x = Product, fill='values')) +
  geom_bar(stat='sum',
    position='dodge') +
  labs(title="Global Top Selling Products - North America vs EU Sales",
       x = "North America Sales in £",
       y = "EU Sales in £") +
  theme_classic()

# Print the plot 
top_sales_by_region

# Specify the ggplotly() function and pass the plot to make interactive
ggplotly(top_sales_by_region) 

# Using melt() to rearrange data 
# To help with visualization of sales data by region side by side
sales_df <- melt(product_sales, id.vars = 'Product', variable.name = 'region')
  
View(sales_df)

# Checking how the top 10 ranked games sell in all regions
# Creating a dataframe of the products by ranking

# Slice to the top 10 ranked products
top_10 <- product_sales[1:10, c(1:4)]

top_10

#Convert product column type to string
top_10$Product = as.character((top_10$Product)) 

#Return the datatype of each column
print(sapply(top_10, class)) 

# Melt the dataframe to help visualizing data by region
top_10_sales <- melt(top_10,  id.vars = 'Product', 
               variable.name = 'region')

# View the dataframe to check it's all ok
View(top_10_sales)


###############################################################################

# 5. Observations and insights
# Your observations and insights here...
  
# While the majority of best selling global products, sell well in both markets 
# there are some of the top selling products sell better either in NA or EU but
# not both markets.  

  
# Correlation plot shows strong positive correlation between the three sales 
# variables. Although Global_Sales is more correlated to NA sales than EU sales
  
  

###############################################################################
###############################################################################

# Week 6 assignment: Making recommendations to the business using R

## The sales department wants to better understand if there is any relationship
## between North America, Europe, and global sales. Therefore, you need to
## investigate any possible relationship(s) in the sales data by creating a 
## simple and multiple linear regression model. Based on the models and your
## previous analysis (Weeks 1-5), you will then provide recommendations to 
## Turtle Games based on:
##   - Do you have confidence in the models based on goodness of fit and
##        accuracy of predictions?
##   - What would your suggestions and recommendations be to the business?
##   - If needed, how would you improve the model(s)?
##   - Explain your answers.

# Instructions
# 1. Load and explore the data.
##  - Continue to use the data frame that you prepared in the Week 5 assignment. 
# 2. Create a simple linear regression model.
##  - Determine the correlation between the sales columns.
##  - View the output.
##  - Create plots to view the linear regression.
# 3. Create a multiple linear regression model
##  - Select only the numeric columns.
##  - Determine the correlation between the sales columns.
##  - View the output.
# 4. Predict global sales based on provided values. Compare your prediction to
#      the observed value(s).
##  - NA_Sales_sum of 34.02 and EU_Sales_sum of 23.80.
##  - NA_Sales_sum of 3.93 and EU_Sales_sum of 1.56.
##  - NA_Sales_sum of 2.73 and EU_Sales_sum of 0.65.
##  - NA_Sales_sum of 2.26 and EU_Sales_sum of 0.97.
##  - NA_Sales_sum of 22.08 and EU_Sales_sum of 0.52.
# 5. Include your insights and observations.

###############################################################################

# 1. Load and explore the data
# View data frame created in Week 5.
View(product_sales)
head(product_sales)

# Determine a summary of the data frame.
summary(product_sales)
dim(product_sales)

###############################################################################

# 2. Create a simple linear regression model
## 2a) Determine the correlation between columns
# Create a linear regression model on the original data.
model_lm <- lm(Global_Sales~EU_Sales,
             data=product_sales)

# View the linear regression model
model_lm
summary(model_lm)

# The linear model explains 72.01% of the data, so while a good indication 
# accuracy could definitely be improved

## 2b) Create a plot (simple linear regression)
# Basic visualisation.

# Plot the residuals
plot(model_lm$residuals)

# We want there be no pattern... 
# Plot the relationship with base R graphics.
plot(product_sales$EU_Sales, product_sales$Global_Sales)
coefficients(model_lm)


# Add line-of-best-fit.
abline(coefficients(model_lm))

###############################################################################

# 3. Create a multiple linear regression model
# Select only numeric columns from the original data frame.
model_m = lm(Global_Sales~NA_Sales+EU_Sales+Product, data = sales_clean)

summary(model_m)

# This is a strong model with all variables having *** and an R-squared value of
# 0.9709

# Multiple linear regression model
# Testing a different version of model to see how the model behaves 

model_m1 = lm(Global_Sales~NA_Sales+EU_Sales, data = product_sales)
summary(model_m1)

# While the variables are remain strong, the R-squared value drops to 0.9664 so
# it's best to use model_m


###############################################################################

# 4. Predictions based on given values
# Compare with observed values for a number of records
predictTest = predict(model_m, newdata = product_sales,
                      interval='confidence')

predictTest

View(predictTest)

# Select Rows by vector of Values
values <- product_sales[product_sales$NA_Sales %in% 
                          c('34.02', '3.93', '2.73','2.26', '22.08' ),]

print(values)

###############################################################################

# 5. Observations and insights
# Your observations and insights here...
# Model_M has a R-squared value of 0.9709, which means it's able to predict
# 97.09 % within a confidence interval band of the Global Sales using NA and 
# EU sales data. This shows a positive relationship between the sales in 
# North Amercia and the EU, as an indication of global sales. 
 


###############################################################################
###############################################################################




