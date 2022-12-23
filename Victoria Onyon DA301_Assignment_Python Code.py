#!/usr/bin/env python
# coding: utf-8

# ### LSE Data Analytics Online Career Accelerator 
# 
# # DA301:  Advanced Analytics for Organisational Impact

# ## Assignment template

# ### Scenario
# You are a data analyst working for Turtle Games, a game manufacturer and retailer. They manufacture and sell their own products, along with sourcing and selling products manufactured by other companies. Their product range includes books, board games, video games and toys. They have a global customer base and have a business objective of improving overall sales performance by utilising customer trends. In particular, Turtle Games wants to understand: 
# - how customers accumulate loyalty points (Week 1)
# - how useful are remuneration and spending scores data (Week 2)
# - can social data (e.g. customer reviews) be used in marketing campaigns (Week 3)
# - what is the impact on sales per product (Week 4)
# - the reliability of the data (e.g. normal distribution, Skewness, Kurtosis) (Week 5)
# - if there is any possible relationship(s) in sales between North America, Europe, and global sales (Week 6).

# # Week 1 assignment: Linear regression using Python
# The marketing department of Turtle Games prefers Python for data analysis. As you are fluent in Python, they asked you to assist with data analysis of social media data. The marketing department wants to better understand how users accumulate loyalty points. Therefore, you need to investigate the possible relationships between the loyalty points, age, remuneration, and spending scores. Note that you will use this data set in future modules as well and it is, therefore, strongly encouraged to first clean the data as per provided guidelines and then save a copy of the clean data for future use.
# 
# ## Instructions
# 1. Load and explore the data.
#     1. Create a new DataFrame (e.g. reviews).
#     2. Sense-check the DataFrame.
#     3. Determine if there are any missing values in the DataFrame.
#     4. Create a summary of the descriptive statistics.
# 2. Remove redundant columns (`language` and `platform`).
# 3. Change column headings to names that are easier to reference (e.g. `renumeration` and `spending_score`).
# 4. Save a copy of the clean DataFrame as a CSV file. Import the file to sense-check.
# 5. Use linear regression and the `statsmodels` functions to evaluate possible linear relationships between loyalty points and age/renumeration/spending scores to determine whether these can be used to predict the loyalty points.
#     1. Specify the independent and dependent variables.
#     2. Create the OLS model.
#     3. Extract the estimated parameters, standard errors, and predicted values.
#     4. Generate the regression table based on the X coefficient and constant values.
#     5. Plot the linear regression and add a regression line.
# 6. Include your insights and observations.

# ## 1. Load and explore the data

# In[183]:


# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from statsmodels.formula.api import ols


# In[184]:


# Load the CSV file(s) as reviews.
reviews = pd.read_csv('turtle_reviews.csv')

# Check the DataFrame:
reviews.info()


# In[185]:


# View the dataframe
reviews.head()


# In[186]:


# Any missing values?
reviews.isna().sum()


# In[187]:


# Explore the data.
print(reviews.shape)
print(reviews.dtypes)


# In[188]:


# Descriptive statistics.
reviews.describe()


# In[189]:


# Checking the number of each gender as categorical variable 
reviews['gender'].value_counts()


# In[190]:


# Checking the number of each education level as categorical variable 
reviews['education'].value_counts()


# In[191]:


# Create a plot of Customer Demographics 
sns.countplot(data=reviews, x='education', hue = 'gender',).set(title = "Turtle Games Customer Demographics",
     xlabel= "Education Level",
     ylabel = "Count of Individuals")
plt.savefig('customer_demographics.png')


# ## 2. Drop columns

# In[192]:


# Drop unnecessary columns - language and platform
reviews.drop(['language', 'platform'], axis=1, inplace=True)

# View column names.
reviews.columns


# ## 3. Rename columns

# In[193]:


# Rename the column headers.
reviews.rename(columns = {'remuneration (k£)':'remuneration', 'spending_score (1-100)':'spending_score'}, inplace = True)

# View column names.
print(reviews.columns)


# ## 4. Save the DataFrame as a CSV file

# In[194]:


# Create a CSV file as output.
reviews.to_csv('reviews.csv')


# In[195]:


# Import new CSV file with Pandas.
import os 
working_directory = os.getcwd()
print(working_directory)


# In[196]:


path = working_directory + '/reviews.csv'
reviews2 = pd.read_csv(path)

# View DataFrame.
reviews2


# ## 5. Linear regression

# In[197]:


# Import the required packages 
# LinearRegression
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
from sklearn import linear_model


# ### 5a) Spending vs Loyalty

# In[198]:


# Independent variable.
x = reviews2['spending_score']

# Dependent variable.
y = reviews2['loyalty_points']

# Check for linearity
plt.scatter(x, y)


# In[199]:


# Run the OLS model on the data.
f = 'y ~ x'

test = ols(f, data = reviews2).fit()

# Print the regression table
test.summary()


# In[200]:


# Checking for homoscedasticity
# Import the necessary library.
import statsmodels.stats.api as sms

# Run the Breusch-Pagan test function on the model residuals and x-variables.
test = sms.het_breuschpagan(test.resid, test.model.exog)

# Print the results of the Breusch-Pagan test.
terms = ['LM stat', 'LM Test p-value', 'F-stat', 'F-test p-value']
print(dict(zip(terms, test)))


# In[201]:


# As the LM Test p-value is less than 0.05 so we can assume homoscedasticity


# In[202]:


# Extract the estimated parameters.
print("Parameters: ", test.params)  

# Extract the standard errors.
print("Standard errors: ", test.bse)  

# Extract the predicted values.
print("Predicted values: ", test.predict())


# In[203]:


# Set the X coefficient and the constant to generate the regression table.
y_pred = (-75.0527) + 33.061693 * x

# View the output
y_pred


# In[204]:


# Plot the graph with a regression line.
# Plot the data points with a scatterplot.
plt.scatter(x, y)

# Plot the regression line (in black)
plt.plot(x, y_pred, color='black')

plt.title("Relationship between Spending Score & Loyalty Points")
plt.xlabel("Spending Score")
plt.ylabel("Loyalty Points")

# Set the x and y limits on the axes.
plt.xlim(0)
plt.ylim(0)

# View the plot.
plt.show()


# ### 5b) renumeration vs loyalty

# In[23]:


# Independent variable.
y = reviews2['loyalty_points'].values.reshape(-1, 1)

# Dependent variable.
x = reviews2['remuneration'].values.reshape(-1, 1)


# In[24]:


# OLS model and summary.
lm = LinearRegression()

# Fit the model.
lm.fit(x, y) 


# In[25]:


# Create  the subset (50/50); 
# Control the shuffling/avoid variation in values between variable.
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.5,
                                                    random_state=100)


# In[26]:


# Predict the training set values.
y_pred = lm.predict(x_train) 

# View the output
y_pred


# In[27]:


# Plot graph with regression line.
# Create a scatterplot with regression line.
plt.scatter(x_train, y_train)  
plt.plot(x_train, y_pred, color = 'green')

# Set the title and legends for the graph.  
plt.title("Relationship between Remuneration and Loyalty Points")
plt.xlabel("Remuneration in £")
plt.ylabel("Number of Loyalty Points") 

# Print the graph. 
plt.show() 


# In[28]:


# Print R-squared value of the test data.
print("R-squared value: ", lm.score(x_train, y_train)) 

# Print the intercept value.
print("Intercept value: ", lm.intercept_) 

# Print the coefficient value.
print("Coefficient value: ", lm.coef_)


# In[29]:


# Predict the test set values.
y_pred_test = lm.predict(x_test) 

# Create a scatterplot with regression line.
plt.scatter(x_test, y_test) 
plt.plot(x_test, y_pred_test, color = 'green')

# Set the title and legends for the graph.  
plt.title("Remuneration vs Loyalty Points")
plt.xlabel("Remuneration")
plt.ylabel("Loyalty Points")

# Print the graph. 
plt.show() 


# ### 5c) age vs loyalty

# In[30]:


# Independent variable.
X = reviews2['age']

# Dependent variable.
y = reviews2['loyalty_points']

# Check for linearity
plt.scatter(X, y)


# In[31]:


# OLS model and summary.
f = 'y ~ X'
test = ols(f, data = reviews2).fit()

test.summary()


# In[32]:


# Extract the estimated parameters.
print("Parameters: ", test.params)  

# Extract the standard errors.
print("Standard errors: ", test.bse)  

# Extract the predicted values.
print("Predicted values: ", test.predict())


# In[33]:


# Set the X coefficient and the constant to generate the regression table.
y_pred = 1736.517719 -4.021805 * X

# View the output.
y_pred


# In[34]:


# Plot graph with regression line.
plt.scatter(X, y)

# Plot the line.
plt.plot(X, y_pred, color='black')

# Set the title and legends for the graph.  
plt.title("Relationship between Age and Loyalty Points")
plt.xlabel("Age of Customers")
plt.ylabel("Number of Loyalty Points") 

# Print the graph. 
plt.show() 

# Save the fig
plt.savefig('age vs loyalty points.png')


# ## 6. Observations and insights

# ***Your observations here...***
# 
# 
# 
# 
# 

# R-squared (R2) is a statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. Using OLS regression analysis, to test how various factors influence the loyalty points (in the case y variable) we are able to see a low to medium correlation between customers' spending score and loyalty points with a R squared value of 0.452 and between remuneration and loyalty points a R squared value of 0.355. 
# The correlation between customers' age and the amount of loyalty points is almost non-existent. 

# # 

# # Week 2 assignment: Clustering with *k*-means using Python
# 
# The marketing department also wants to better understand the usefulness of renumeration and spending scores but do not know where to begin. You are tasked to identify groups within the customer base that can be used to target specific market segments. Use *k*-means clustering to identify the optimal number of clusters and then apply and plot the data using the created segments.
# 
# ## Instructions
# 1. Prepare the data for clustering. 
#     1. Import the CSV file you have prepared in Week 1.
#     2. Create a new DataFrame (e.g. `df2`) containing the `renumeration` and `spending_score` columns.
#     3. Explore the new DataFrame. 
# 2. Plot the renumeration versus spending score.
#     1. Create a scatterplot.
#     2. Create a pairplot.
# 3. Use the Silhouette and Elbow methods to determine the optimal number of clusters for *k*-means clustering.
#     1. Plot both methods and explain how you determine the number of clusters to use.
#     2. Add titles and legends to the plot.
# 4. Evaluate the usefulness of at least three values for *k* based on insights from the Elbow and Silhoutte methods.
#     1. Plot the predicted *k*-means.
#     2. Explain which value might give you the best clustering.
# 5. Fit a final model using your selected value for *k*.
#     1. Justify your selection and comment on the respective cluster sizes of your final solution.
#     2. Check the number of observations per predicted class.
# 6. Plot the clusters and interpret the model.

# ## 1. Load and explore the data

# In[36]:


# Import necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist

import warnings
warnings.filterwarnings('ignore')


# In[37]:


# Load the CSV file(s) as df2.
path = working_directory + '/reviews.csv'
df = pd.read_csv(path)

# View DataFrame.
df


# In[38]:


# Drop unnecessary columns.
df2 = df.filter(['remuneration', 'spending_score'], axis=1)
         
# View DataFrame.
df2


# In[39]:


# Explore the data.
print(df2.shape)
print(df2.dtypes)


# In[40]:


# Descriptive statistics.
df2.describe


# ## 2. Plot

# In[41]:


# Create a scatterplot with Seaborn to visualise the data.
sns.scatterplot(x='remuneration',
                y='spending_score',
                data=df2)


# In[42]:


# Create a pairplot with Seaborn to visualise the distribution of the data.
x = df[['remuneration', 'spending_score']]

sns.pairplot(df2,
             vars=x,
             diag_kind= 'kde')


# ## 3. Elbow and silhoutte methods

# In[43]:


# Determine the number of clusters: Elbow method.
from sklearn.cluster import KMeans

# Elbow chart for us to decide on the number of optimal clusters.
# Create an empty list to store different cluster sizes

ss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i,
                    init = 'k-means++', 
                    max_iter = 500,
                    n_init = 10,
                    random_state = 42)
    kmeans.fit(x)
    ss.append(kmeans.inertia_)

plt.plot(range(1, 11),
         ss,
         marker='o')

plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("SS")

plt.show()


# In[44]:


# Determine the number of clusters: Silhouette method.
from sklearn.metrics import silhouette_score

# Find the range of clusteres to be used using silhouette method
sil = []
kmax = 10

for k in range(2, kmax+1):
    kmeans_s = KMeans(n_clusters = k).fit(x)
    labels = kmeans_s.labels_
    sil.append(silhouette_score(x,
                               labels,
                               metric = 'euclidean'))
    
# Plot the silhouette method
plt.plot(range(2, kmax+1),
        sil,
        marker='o')

plt.title("The Silhouette Method")
plt.xlabel("Number of clusters")
plt.ylabel("Sil")

plt.show()


# ## 4. Evaluate k-means model at different values of *k*

# In[45]:


# Using five clusters to evaluate and fit the model

kmeans = KMeans(n_clusters = 5,
               max_iter = 15000,
               init='k-means++',
               random_state=42).fit(x)

clusters = kmeans.labels_
x['K-Means Predicted'] = clusters

sns.pairplot(x,
            hue='K-Means Predicted',
            diag_kind='kde')

plt.xlabel("Number of clusters")
plt.ylabel("SS")


# In[46]:


# Using four clusters as part of the silhouette model to evaluate and fit the model

kmeans = KMeans(n_clusters = 4,
               max_iter = 15000,
               init='k-means++',
               random_state=42).fit(x)

clusters = kmeans.labels_
x['K-Means Predicted'] = clusters

sns.pairplot(x,
            hue='K-Means Predicted',
            diag_kind='kde')

plt.xlabel("Number of clusters")
plt.ylabel("SS")


# In[47]:


# Using six clusters as part of the silhouette model to evaluate and fit the model

kmeans = KMeans(n_clusters = 6,
               max_iter = 15000,
               init='k-means++',
               random_state=42).fit(x)

clusters = kmeans.labels_
x['K-Means Predicted'] = clusters

sns.pairplot(x,
            hue='K-Means Predicted',
            diag_kind='kde')

plt.xlabel("Number of clusters")
plt.ylabel("SS")


# ## 5. Fit final model and justify your choice

# Based on the k-means analysis it is recommended to group the Turtle Games customers into 4 clusters, as this will enable four clear distinct market segments (or 'hard' clustering). This is useful approach to this business problem, as each customer will be clearly assigned to one market segmentation and can an appropiate marketing campagin can be used.  

# In[48]:


#Apply the final model.
# Chose  4 clustters 
kmeans = KMeans(n_clusters = 4,
               max_iter = 15000,
               init='k-means++',
               random_state=42).fit(x)

clusters = kmeans.labels_
x['K-Means Predicted'] = clusters


# In[49]:


# Check the number of observations per predicted class.
x['K-Means Predicted'].value_counts()


# In[50]:


# View the head of the K-means predicted
print(x.head())


# ## 6. Plot and interpret the clusters

# In[51]:


# Visualising the clusters.
# Set plot size.
sns.set(rc = {'figure.figsize':(12, 8)})

# Create the scatterplot
kmeans = sns.scatterplot(x='remuneration',
                y='spending_score',
                data=x , hue='K-Means Predicted',
                palette=['red', 'green', 'blue', 'black'])

kmeans.set_title("Customer Segmentation by Spending Score and Remuneration", fontsize=20)
kmeans.set_xlabel("Remuneration in £", fontsize=14)
kmeans.set_ylabel("Spending Score", fontsize=14)

plt.savefig('K-Means.png')


# ## 7. Discuss: Insights and observations
# 
# ***Your observations here...***
# Using k-means clustering to group customers by simiarilities in remuneration and spending scores, we can identify four distinct groups within the Turtle Games customer database. This means each customer is assigned to a segmentation and will assist with the development of tailored marketing and sales strategies. 
# 
# The majority of customers 1013 individuals or 50.6% fall into one cluster (cluster 0), which have high remuneration but low spending scores. The customers who have high spending scores are fairly spilt with 356(17.8%) customers with low remuneration and 351(17.55%) customers with high remuneration. The remaining 14% is low remuneration and low spending score customers. 
# The recommendation is to focus marketing and sales campagin on increasing the spend of the customers in cluster 0 and encouraging ongoing sales from the customers with high spending scores. Further investigation into the similarities between the two clusters with high spending scores may be very useful. 

# # 

# # Week 3 assignment: NLP using Python
# Customer reviews were downloaded from the website of Turtle Games. This data will be used to steer the marketing department on how to approach future campaigns. Therefore, the marketing department asked you to identify the 15 most common words used in online product reviews. They also want to have a list of the top 20 positive and negative reviews received from the website. Therefore, you need to apply NLP on the data set.
# 
# ## Instructions
# 1. Load and explore the data. 
#     1. Sense-check the DataFrame.
#     2. You only need to retain the `review` and `summary` columns.
#     3. Determine if there are any missing values.
# 2. Prepare the data for NLP
#     1. Change to lower case and join the elements in each of the columns respectively (`review` and `summary`).
#     2. Replace punctuation in each of the columns respectively (`review` and `summary`).
#     3. Drop duplicates in both columns (`review` and `summary`).
# 3. Tokenise and create wordclouds for the respective columns (separately).
#     1. Create a copy of the DataFrame.
#     2. Apply tokenisation on both columns.
#     3. Create and plot a wordcloud image.
# 4. Frequency distribution and polarity.
#     1. Create frequency distribution.
#     2. Remove alphanumeric characters and stopwords.
#     3. Create wordcloud without stopwords.
#     4. Identify 15 most common words and polarity.
# 5. Review polarity and sentiment.
#     1. Plot histograms of polarity (use 15 bins) for both columns.
#     2. Review the sentiment scores for the respective columns.
# 6. Identify and print the top 20 positive and negative reviews and summaries respectively.
# 7. Include your insights and observations.

# ## 1. Load and explore the data

# In[52]:


# Import all the necessary packages.
import pandas as pd
import numpy as np
import nltk 
import os 
import matplotlib.pyplot as plt

# nltk.download ('punkt').
# nltk.download ('stopwords').

from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from textblob import TextBlob
from scipy.stats import norm

# Import Counter.
from collections import Counter

import warnings
warnings.filterwarnings('ignore')


# In[53]:


# Load the data set as df3.
path = working_directory + '/reviews.csv'
df3 = pd.read_csv(path)

# View DataFrame.
df3.head()


# In[54]:


# Explore data set.
print(df3.shape)
print(df3.dtypes)
df3.describe


# In[55]:


# Keep necessary columns. Drop unnecessary columns.
df3 = df.filter(['review', 'summary'], axis=1)

# View DataFrame.
df3


# In[56]:


# Determine if there are any missing values.
# All rows and columns have data 
df3.isnull()


# ## 2. Prepare the data for NLP
# ### 2a) Change to lower case and join the elements in each of the columns respectively (review and summary)

# In[57]:


# Review: Change all to lower case and join with a space.
df3['review']= df3['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df3.head()


# In[58]:


# Summary: Change all to lower case and join with a space.
df3['summary']= df3['summary'].apply(lambda x: " ".join(x.lower() for x in x.split()))

df3.head()


# ### 2b) Replace punctuation in each of the columns respectively (review and summary)

# In[59]:


# Replace all the punctuations in review column.
# Remove punctation and replace with blank spaces
df3['review'] = df3['review'].str.replace('[^\w\s]', '')

# View output.
df3.head()


# In[60]:


# Replace all the puncuations in summary column.

df3['summary'] = df3['summary'].str.replace('[^\w\s]', '')

# View output.
df3.head()


# ### 2c) Drop duplicates in both columns

# In[93]:


# Check the number of duplicates
df3.duplicated().sum()

# Drop duplicates in both columns.
reviews_text = df.drop_duplicates()
reviews_text.reset_index(inplace=True)

# View DataFrame.
reviews_text.tail()


# ## 3. Tokenise and create wordclouds

# In[94]:


# Import the nltk libraries to tokenize the DataFrame
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize


# In[97]:


# Tokensize the reviews columns
reviews_text['tokens'] = reviews_text['review'].apply(word_tokenize)

# Preview the result
reviews_text['tokens'].head()


# In[99]:


# Create a empty string variable to collect all of the text from reviews column.
review_comments = []
for i in range(reviews_text.shape[0]):
    # Add each comment
    review_comments = review_comments + reviews_text['tokens'][i]


# In[100]:


review_comments


# In[101]:


# Tokensize the summary columns
reviews_text['summary_tokens'] = reviews_text['summary'].apply(word_tokenize)

# Preview the result
reviews_text['summary'].head()


# In[102]:


# Create a empty string variable to collect all of the text from summary column.
summary_comments = ''
for i in range(reviews_text.shape[0]):
    # Add each comment
    summary_comments = summary_comments + reviews_text['summary'][i]    


# In[103]:


# Review: Create a word cloud.
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


# In[104]:


# Review: Plot the WordCloud image.
sns.set(color_codes=True)
reviews_word_cloud = WordCloud(width = 1600, height = 900,
                      background_color ='white',
                      colormap ='plasma',
                      stopwords = 'none',
                      min_font_size = 10).generate(review_comments)

plt.figure(figsize = (16, 9), facecolor = None)
plt.imshow(reviews_word_cloud)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# In[105]:


# Summary: Create a word cloud.
# Review: Plot the WordCloud image.
sns.set(color_codes=True)
summary_word_cloud = WordCloud(width = 1600, height = 900,
                      background_color ='white',
                      colormap ='plasma',
                      stopwords = 'none',
                      min_font_size = 10).generate(summary_comments)


# In[106]:


# Summary: Plot the WordCloud image.
plt.figure(figsize = (16, 9), facecolor = None)
plt.imshow(summary_word_cloud)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()


# ## 4. Frequency distribution and polarity
# ### 4a) Create frequency distribution

# In[109]:


# Determine the frequency distribution.
from nltk.probability import FreqDist

# Calculate the frequency distribution
fdist = FreqDist(review_comments)
fdist


# ### 4b) Remove stopwords

# In[119]:


# Download the stop word list
nltk.download('stopwords')
from nltk.corpus import stopwords

# Create a set of English stop words
english_stopwords = set(stopwords.words('english'))

# Create a filtered list of review tokens without stop words
review_tokens = [x for x in review_comments if x.lower() not in english_stopwords]

# Deine an empty string variable
review_tokens_string = ''
for value in review_tokens:
    # Add each filtered token word to the string
    review_tokens_string = review_tokens_string + value + ' '


# In[124]:


# Create a filtered list of summary tokens without stop words
summary_tokens = [x for x in summary_comments if x.lower() not in english_stopwords]

# Deine an empty string variable
summary_tokens_string = ''
for value in review_tokens:
    # Add each filtered token word to the string
    summary_tokens_string = summary_tokens_string + value + ' '


# In[128]:


# Combine the summary and review token strings together
all_tokens = summary_tokens_string + review_tokens_string
all_tokens


# ### 4c) Create wordcloud without stopwords

# In[130]:


# Create a wordcloud without stop words.
# Review: Plot the WordCloud image.
sns.set(color_codes=True)
all_reviews_word_cloud = WordCloud(width = 1600, height = 900,
                      background_color ='white',
                      colormap ='plasma',
                      min_font_size = 10).generate(all_tokens)


# In[170]:


# Plot the wordcloud image.
word_cloud = plt.figure(figsize = (16, 9), facecolor = None)
plt.imshow(all_reviews_word_cloud)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()
word_cloud.savefig('wordcloud.jpg')


# ### 4d) Identify 15 most common words and polarity

# In[171]:


# Determine the 15 most common words.
from collections import Counter

# Generate a DataFrame from counter
counts = pd.DataFrame(Counter(all_tokens).most_common(15),
                     columns=['Word','Frequency']).set_index('Word')
# Display the result
counts


# ## 5. Review polarity and sentiment: Plot histograms of polarity (use 15 bins) and sentiment scores for the respective columns.

# In[141]:


# Provided function.
def generate_polarity(comment):
    '''Extract polarity score (-1 to +1) for each comment'''
    return TextBlob(comment).sentiment[0]


# In[144]:


# Populate a new column with polarity scores for each review
reviews_text['polarity'] = reviews_text['review'].apply(generate_polarity)

reviews_text['polarity'].head()


# In[176]:


# Review: Create a histogram plot with bins = 15.
# Histogram of polarity
# Set the number of bins
num_bins = 15

# Set the plot area
review_sentiment = plt.figure(figsize=(16, 9))

# Define the bars
n, bins, patches = plt.hist(reviews_text['polarity'], num_bins, facecolor='red', alpha=0.6)

# Set the labels
plt.xlabel('Polarity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Histogram of review sentiment score polarity", fontsize=20)

# Histogram of sentiment score
plt.show()

review_sentiment.savefig('review_sentiment.jpg')


# In[150]:


# Populate a new column with polarity scores for each review
reviews_text['summary_polarity'] = reviews_text['summary'].apply(generate_polarity)

reviews_text['summary_polarity'].head()


# In[175]:


# Review: Create a histogram plot with bins = 15.
# Histogram of polarity
# Set the number of bins
num_bins = 15

# Set the plot area
summary_sentiment = plt.figure(figsize=(16, 9))

# Define the bars
n, bins, patches = plt.hist(reviews_text['summary_polarity'], num_bins, facecolor='blue', alpha=0.6)

# Set the labels
plt.xlabel('Polarity', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title("Histogram of summary sentiment score polarity", fontsize=20)

# Histogram of sentiment score
plt.show()
summary_sentiment.savefig('summary_sentiment.jpg')


# ## 6. Identify top 20 positive and negative reviews and summaries respectively

# In[181]:


# Top 20 negative reviews.
reviews_negative_sentiment = reviews_text.nsmallest(20, 'polarity')

# Adjust the column width
reviews_negative_sentiment.style.set_properties(subset=['COMMENTS'], **{'width': '1200px'})

# View output.
reviews_negative_sentiment


# In[182]:


# Top 20 negative summaries.
summary_negative_sentiment = reviews_text.nsmallest(20, 'summary_polarity')

# Adjust the column width
summary_negative_sentiment.style.set_properties(subset=['COMMENTS'], **{'width': '1200px'})

# View output.
summary_negative_sentiment


# In[163]:


# Top 20 positive reviews.
review_positive_sentiment = reviews_text.nlargest(20, 'polarity')

# Adjust the column width
review_positive_sentiment.style.set_properties(subset=['COMMENTS'], **{'width': '1200px'})

# View output.
review_positive_sentiment


# In[164]:


# Top 20 positive summaries.
summary_positive_sentiment = reviews_text.nlargest(20, 'summary_polarity')

# Adjust the column width
summary_positive_sentiment.style.set_properties(subset=['COMMENTS'], **{'width': '1200px'})

# View output.
summary_positive_sentiment


# ## 7. Discuss: Insights and observations
# 
# ***Your observations here...***

# The sentiment analysis of the customer reviews of Turtle Games showes slightly positive skewed sentiment, which is a great result for a company. 

# # 
