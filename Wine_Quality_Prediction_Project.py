#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy
import os
import matplotlib.pyplot as plot
import seaborn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

# ### Load Dataset

# In[36]:

def load_csv(file_name):
    """ function that loads csv from local file
    return csv """
    return pandas.read_csv(file_name)


# In[37]:


file_name = 'data' + os.sep + 'WineQualityMix.csv'
data_frame = load_csv(file_name)
data_frame


# In[38]:


print(data_frame.info())
print('------------------------------------')

# In[39]:


print(data_frame.describe())
print('------------------------------------')

# # # Visualization
# # 
# # 
# # 

#Box Plot:

#seaborn.boxplot(data_frame)

# # In[40]:

# # ### Histogram 

#data_frame.hist(bins=100,figsize=(20,20))
#plot.show()


# # In[41]:


# plot.figure(figsize=[5,3])
# # plot bar graph
# plot.bar(data_frame['quality'],data_frame['alcohol'],color='gray')
# # label x-axis
# plot.xlabel('quality')
# # label y-axis
# plot.ylabel('alcohol')
# plot.show()

# # # Correlation
# # 
# # 
# # 
# # ### # correlation between every two features

# # In[42]:


#plot.figure(figsize=[19,10],facecolor='white')
#seaborn.heatmap(data_frame.corr(),annot=True)
#plot.show()

# # # DELETE!!!
# # 
# # ### Now, we have to find those features that are fully correlated to each other by this we reduce the number of features from the data.
# # 
# # ### If you think that why we have to discard those correlated, because relationship among them is equal they equally impact on model accuracy so, we delete one of them.

# # In[43]:


# loop for columns
for column_index in range(len(data_frame.corr().columns)):
    # loop for rows
    for row_index in range(column_index):
        if abs(data_frame.corr().iloc[column_index, row_index]) >0.7:
            name = data_frame.corr().columns[row_index]
            print(name)


# # ### Here we write a python program with that we find those features whose correlation number is high, as you see in the program we set the correlation number greater than 0.7 it means if any feature has a correlation value above 0.7 then it was considered as a fully correlated feature, at last, we find the feature total sulfur dioxide which satisfy the condition. So, we drop that feature

# # In[44]:


new_data_frame=data_frame.drop('total sulfur dioxide',axis=1)


# # # Handle null values

# # In[45]:


new_data_frame.isnull().sum()


# # In[46]:


# using 'numeric_only=True' we handle only numerical variables value and ignore strings

new_data_frame.update(new_data_frame.fillna(new_data_frame.mean(numeric_only=True)))
new_data_frame


# # ### Filling null values with the fillna() function.

# # In[47]:


next_data_frame = pandas.get_dummies(new_data_frame,drop_first=True)
# display new dataframe
next_data_frame


# # ### get_dummies() function which is used for handling categorical columns (with string values), in our dataset ‘Type’ column feature contains two types 'Red' and 'White', where Red consider as 0 and white considers as 1

# # In[48]:

next_data_frame['best quality'] = [1 if quality >=7 else 0 for quality in data_frame.quality]
print(next_data_frame)



# # # Splitting dataset

# # In[50]

##################### Jony  #########################

#X =  next_data_frame.iloc[:, [2,3]]
#y =  next_data_frame['best quality']
#x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=40)

##################### Jony  #########################


# independent variables
x = next_data_frame.drop(['quality','best quality'],axis=1)
# dependent variable
y = next_data_frame['best quality']
# # (alredy imported) --> train_test_split()
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=40)


# # # Normalization

# # In[ ]:


# creating scale object 
norm = MinMaxScaler()
# fit the new scalde data
norm_fit = norm.fit(x_train)
# transformation of training data
new_x_train = norm_fit.transform(x_train)
# transformation of testing data
new_x_test = norm_fit.transform(x_test)
# display values
print(new_x_train)


# # # Applying Model
# # 
# # This is the last step where we apply any suitable model which will give more accuracy, here we will use RandomForestClassifier because it was the only ML model that gives the 88% accuracy which was considered as the best accuracy.

# # # RandomForestClassifier:-

# # In[ ]:


# importing modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# for error checking
from sklearn.metrics import mean_squared_error
#creating RandomForestClassifier constructor
rnd = RandomForestClassifier()
# fit data
fit_rnd = rnd.fit(new_x_train,y_train)
# predicting score
rnd_score = rnd.score(new_x_test,y_test)
print('score of Random Forest Classifier model is : ',rnd_score)
# display error rate
print('calculating the error')
# calculating mean squared error
rnd_MSE = mean_squared_error(y_test, y_predict)
# calculating root mean squared error
rnd_RMSE = numpy.sqrt(rnd_MSE)
# display MSE
print('mean squared error is : ',rnd_MSE)
# display RMSE
print('root mean squared error is : ',rnd_RMSE)
print(classification_report(x_predict,y_test))


# # Now, we are at the end of our article, we can differentiate the predicted values and actual value.

# # In[ ]:


# x_predict = list(rnd.predict(x_test))
# predicted_data_frame = {'predicted_values': x_predict, 'original_values': y_test}
# #creating new dataframe
# pandas.DataFrame(predicted_data_frame).head(20)


# # # Saving Model
# # At last, we save our machine learning model:

# # In[ ]:


# import pickle
# file = 'wine_quality_save'
# #save file
# save = pickle.dump(rnd,open(file,'wb'))

