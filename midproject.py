#!/usr/bin/env python
# coding: utf-8

# ## This Project goal is to identify the Risk of a person getting a heart attack using a data collected collected from the patients

# ## Import The Libraries

# In[77]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings


# ### Load The Dataset

# In[81]:


TRAINDATAPATH = os.path.join(os.getcwd(), 'heart.csv')
df=pd.read_csv(TRAIN_DATA_PATH)

df.head()


# ### Check The Datatypes of the dataset

# In[82]:


df.info()


# ### So The Data is Numeric Without Categorial Features maybe we can add some later

# ### Let's Check The Unique Values in each features

# In[83]:


df.nunique()


# ### Check The Data For Nan Values or '0' Values our Duplicated Values

# In[84]:


df.duplicated().sum() 


# In[85]:


df.drop_duplicates(inplace=True)


# In[86]:


df.describe()


# ### The Data is now clean with no non-meaningful values '0' or Nans

# ### Now Lets start with Univariate Analysis

# ### -Distrubtion of the Sex Feature

# In[89]:


labels = ['Males', 'Females']
sns.set(style="darkgrid")
plt.figure(figsize=(6, 6))
sns.color_palette("pastel")
plt.pie(df['sex'].value_counts(), labels=labels, autopct='%1.1f%%', startangle=90, shadow=False, colors=sns.color_palette("Set2"))
plt.title('Gender Distribution')
plt.axis('equal') 
plt.show()


# ### Distrubtion of Chest Pain 'cp' Now i will make a Dict to map The Labels To The correct Values
# 

# In[91]:


cp_labels = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}

plt.figure(figsize=(8, 6))
sns.set(style='whitegrid')
sns.countplot(x=df['cp'].map(cp_labels), data=df, palette="Set1")

plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.title('Chest Pain Type Distribution')
plt.xticks(rotation=45)

plt.show()


# # Dist of Fasting Blood Sugar

# In[92]:


fbslabels = {0: 'Fasting Blood Sugar < 120 mg/dl', 1: 'Fasting Blood Sugar > 120 mg/dl'}
plt.figure(figsize=(6, 6))
sns.set(style="whitegrid")
sns.countplot(x=df['fbs'].map(fbs_labels), data=df, palette="Set2")
plt.xlabel('Fasting Blood Sugar Levels')
plt.ylabel('Counts')
plt.title('Fasting Blood Sugar Distribution')
plt.xticks(rotation=45)
plt.show()


# ### Dist of Resting ECG Results

# In[93]:


restecg_labels = {0: 'Normal', 1: 'ST-T wave normality', 2: 'Left ventricular hypertrophy'}
plt.figure(figsize=(8, 6))
sns.set(style="darkgrid")
sns.countplot(x=df['restecg'].map(restecg_labels), data=df, palette="Set3")
plt.xlabel('Resting ECG Results')
plt.ylabel('Count')
plt.title('Resting ECG Results Distribution')
plt.show()


# ### Dist of NO.of Major Vessels

# In[95]:


plt.figure(figsize=(8, 8))
sns.set(style="darkgrid")
sns.countplot(x='caa', data=df, palette="Set3")
plt.xlabel('Number of Major Vessels Colored by Fluoroscopy')
plt.ylabel('Count')
plt.title('Distribution of Major Vessels Colored by Fluoroscopy')
plt.show()


# ### -Dist of Thallium Stress

# In[96]:


plt.figure(figsize=(8, 7))
sns.set(style="darkgrid")
sns.countplot(x='thall', data=df, palette="Set3")
plt.xlabel('Thallium Stress Test Result')
plt.ylabel('Count')
plt.title('Distribution of Thallium Stress Test Results')
plt.show()


# ### Disturbtion of Exng

# In[97]:


exng_labels = {0: 'No', 1: 'Yes'}
plt.figure(figsize=(8, 8))
sns.set(style="darkgrid")
sns.countplot(x=df['exng'].map(exng_labels) ,data=df, palette="Set3")
plt.xlabel('Exercise Induced Angina')
plt.ylabel('Heart Disease Presence')
plt.title('Distribution of Exercise Induced Angina and Heart Disease')
plt.show()



# ### Dist of SLP

# In[99]:


plt.figure(figsize=(9, 7))
sns.set(style="darkgrid")
sns.countplot(x='slp', data=df, palette="Set3")
plt.xlabel('Slope of Peak Exercise ST Segment')
plt.ylabel('Count')
plt.title('Distribution of Slope of Peak Exercise ST Segment')
plt.show()


# ###  The Continous Features

# In[100]:


plt.figure(figsize=(9, 8))
sns.distplot(df['age'], kde=True, bins=20, color="red")
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()


# In[101]:


plt.figure(figsize=(10, 6))
sns.set(style="darkgrid")
sns.violinplot(x='trtbps', data=df, palette="husl", inner="quartile")
plt.xlabel('Resting Blood Pressure (mm Hg)')
plt.title('Distribution of Resting Blood Pressure')
plt.show()


# ## Distribution of Chrol

# In[105]:


plt.figure(figsize=(10, 8))
sns.set(style="darkgrid")
sns.boxplot(x='chol', data=df, palette="Set3")
plt.xlabel('Serum Cholesterol Level (mg/dl)')
plt.title('Distribution of Serum Cholesterol Level')

plt.show()


# ### Distribution of Thalachh

# In[107]:


plt.figure(figsize=(10, 8))
sns.set(style="darkgrid")
sns.boxplot(x='thalachh', data=df, palette="Set3")

plt.xlabel('Maximum Heart Rate (Thalachh)')
plt.title('Distribution of Maximum Heart Rate (Thalachh)')

plt.show()


# ### Distribution of old peak

# In[109]:


plt.figure(figsize=(10, 8))
sns.set(style="darkgrid")
sns.swarmplot(x='oldpeak', data=df, palette="Set3")
plt.xlabel('ST Depression (Oldpeak)')
plt.title('Distribution of ST Depression')

plt.show()


# ### Distribution of Target

# In[111]:


plt.figure(figsize=(8, 8))
sns.set(style="darkgrid")
sns.countplot(x='output', data=df, palette="Set3")
plt.xticks(ticks=[0, 1], labels=['No', 'Yes'])
plt.xlabel('Heart Disease Presence')
plt.ylabel('Count')
plt.title('Distribution of Heart Disease Presence')
plt.show()


# ## -Age vs Max Heart Rate with Respect to Output

# In[112]:


plt.figure(figsize=(10, 8))
sns.scatterplot(x='age', y='thalachh', data=df, hue='output', palette='Set1')
plt.xlabel('Age')
plt.ylabel('Maximum Heart Rate (Thalachh)')
plt.title('Scatter Plot: Age vs. Maximum Heart Rate')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()


# ### Pressure Vs Cholestrol

# In[30]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='trtbps', y='chol', data=df, hue='output', palette='Set1')
plt.xlabel('Resting Blood Pressure (trtbps)')
plt.ylabel('Serum Cholesterol Level (chol)')
plt.title('Scatter Plot: Resting Blood Pressure vs. Serum Cholesterol')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()


# ### Output vs cpa

# In[31]:


plt.figure(figsize=(8, 6))
sns.barplot(x='cp', y='output', data=df, palette='Set1')
plt.xlabel('Chest Pain Type (cp)')
plt.ylabel('Proportion of Heart Disease Presence')
plt.title('Heart Disease Presence by Chest Pain Type')
plt.show()


# ### Resting Rate vs output

# In[32]:


plt.figure(figsize=(8, 6))
sns.barplot(x='restecg', y='output', data=df, palette='Set1')
plt.xlabel('Resting Electrocardiographic Results (restecg)')
plt.ylabel('Proportion of Heart Disease Presence')
plt.title('Heart Disease Presence by Resting Electrocardiographic Results')
plt.show()


# ### No of Vessels vs output

# In[33]:


plt.figure(figsize=(8, 6))
sns.barplot(x='caa', y='output', data=df, palette='Set1')
plt.xlabel('Number of Major Vessels Colored by Fluoroscopy (caa)')
plt.ylabel('Proportion of Heart Disease Presence')
plt.title('Heart Disease Presence by Number of Major Vessels Colored by Fluoroscopy')
plt.show()


# ### thall vs output

# In[34]:


plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
sns.countplot(x='thall', hue='output', data=df, palette='Set1')
plt.xlabel('Thallium Stress Test Result (thall)')
plt.ylabel('Count')
plt.title('Heart Disease Presence by Thallium Stress Test Result')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()


# ### EXNG vs output

# In[44]:


plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
sns.countplot(x='exng', hue='output', data=df, palette='Set1')
plt.xlabel('Exercise-Induced Angina (exng)')
plt.ylabel('Count')
plt.title('Heart Disease Presence by Exercise-Induced Angina')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()


# ### Sex vs output
# 

# In[73]:


plt.figure(figsize=(8, 6))
sns.set(style="whitegrid")
sns.countplot(x='sex', hue='output', data=df, palette='Set1')
plt.xlabel('Gender (0 = Female, 1 = Male)')
plt.ylabel('Count')
plt.title('Heart Disease Presence by Gender')
plt.legend(title='Heart Disease', labels=['No', 'Yes'])
plt.show()


# In[79]:


sns.pairplot(df, hue='output', palette=["#8000ff", "#da8829"], diag_kws={"bw": 0.2}) 
plt.show()


# In[80]:


correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[ ]:




