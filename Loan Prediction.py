
# importing libraries

# import pandas for importing csv files
import pandas as pd

#import numpy for working with arrays
import numpy as np


#################################################

# creating a data frame using CSV files

df=pd.read_csv("loan_data.csv")
print(df.head())


#################################################

# dropping the irrelevant columns 'Loan_ID','Gender','Property_Area'

df=df.drop(['Loan_ID','Gender','Property_Area'],axis=1)
print(df.head())

# displaying the number of columns

cols = df.shape[1]
print("Columns: " + str(cols))


#################################################

#dropping irrelevant columns 'Married','Education'

df=df.drop(['Married','Education'],axis=1)
print(df.head())


#displaying the number of columns

cols = df.shape[1]
print("Columns: " + str(cols))


#################################################

# cleaning categorical values in 'Self_Employed'

# 2 columns 'Yes' & 'No'
dummy =pd.get_dummies(df['Self_Employed'])
# print(dummy.head())

# connecting the 2 columns by the dataset
df= pd.concat((df,dummy),axis=1)
#print(df.head())

# dropping column 'No' 
df=df.drop(['Self_Employed'],axis=1)
df=df.drop(['No'],axis=1)
df=df.rename(columns={"Yes":"Self_Employed"})
print(df.head())


#################################################

# cleaning categorical values in 'Loan_Status'

# 2 columns 'Y' & 'N'
dummy =pd.get_dummies(df['Loan_Status'])
# print(dummy.head())

# connecting the 2 columns by the dataset
df= pd.concat((df,dummy),axis=1)
#print(df.head())

# dropping column 'N' 
df=df.drop(['Loan_Status'],axis=1)
df=df.drop(['N'],axis=1)
df=df.rename(columns={"Y":"Loan_Status"})
print(df.head())


#################################################

# cleaning categorical values in 'Married'

# 2 columns 'Yes' & 'No'
#dummy =pd.get_dummies(df['Married'])
#print(dummy.head())

# connecting the 2 columns by the dataset
#df= pd.concat((df,dummy),axis=1)
#print(df.head())

# dropping column 'No' 
#df=df.drop(['Married'],axis=1)
#df=df.drop(['No'],axis=1)
#df=df.rename(columns={"Yes":"Married"})
#print(df.head())

#################################################

# cleaning categorical values in 'Education'

# 2 columns 'Graduate' & 'Not Graduate'
#dummy =pd.get_dummies(df['Education'])
#print(dummy.head())

# connecting the 2 columns by the dataset
#df= pd.concat((df,dummy),axis=1)
#print(df.head())

# dropping column 'Not Graduate' 
#df=df.drop(['Education'],axis=1)
#df=df.drop(['Not Graduate'],axis=1)
#df=df.rename(columns={"Graduate":"Education"})
#print(df.head())

#################################################

# cleaning the nulls and missing values

# remove rows that contain nulls
# print (df.isna().any())
# df=df.dropna()

# test the number of nulls in every column
# print (df.isna().any())
# print(df.head())

# fetching the number of rows 
# rows=df.shape[0]

# displaying the number of rows 
# print("Rows :" +str(rows))

#################################################
# cleaning inconsistent datatypes

df ['Dependents'] = df ['Dependents'].replace (['3+'],'3')

#################################################
# fill the null values with mode 

df["Self_Employed"]= df["Self_Employed"].fillna(df["Self_Employed"].mode())

df["Loan_Status"]= df["Loan_Status"].fillna(df["Loan_Status"].mode())

#df["Married"]= df["Married"].fillna(df["Married"].mode())

#df["Education"]= df["Education"].fillna(df["Education"].mode())

df["Dependents"]= df["Dependents"].fillna(df["Dependents"].mode()[0])

df["Credit_History"]= df["Credit_History"].fillna(df["Credit_History"].mode()[0])


#################################################

# Box Plot without dropping outliers

#import seaborn as sns
#sns.boxplot(df['LoanAmount'])
#sns.boxplot(df['Loan_Amount_Term'])


#################################################
# fill the null values with median if outliers exist

#df["LoanAmount"]= df["LoanAmount"].fillna(df["LoanAmount"].median())

#df["Loan_Amount_Term"]= df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median())


#################################################

# Box Plot with dropping

import seaborn as sns
sns.boxplot(df['ApplicantIncome'])

# Position of the Outlier
print(np.where(df['ApplicantIncome']>10000))


sns.boxplot(df['CoapplicantIncome'])

# Position of the Outlier
print(np.where(df['CoapplicantIncome']>5000))


sns.boxplot(df['LoanAmount'])

# Position of the Outlier
print(np.where(df['LoanAmount']>300))


sns.boxplot(df['Loan_Amount_Term'])

# Position of the Outlier
print(np.where(df['Loan_Amount_Term']>400))


# Outlier’s Index of column 'ApplicantIncome'
df.drop([9,34,  54,  67, 102, 106, 115, 119, 126, 128, 130, 138, 144, 146,155, 171, 183, 185, 191, 199, 254, 258, 271, 278, 284, 308, 324,
        333, 369, 370, 409, 424, 432, 435, 438, 443, 467, 474, 477, 482,
        486, 492, 505, 508, 524, 532, 533, 556, 560, 571, 593, 603], inplace = True)
 
#Outlier’s Index of column 'CoapplicantIncome'
 
df.drop([ 12, 21, 38,  91, 122, 135, 159, 173,177,180, 181, 188,
       242, 253 ,349,372,  402, 417, 440, 444, 501, 502,512,522
       ,529, 580,599], inplace = True)

# Outlier’s Index of column 'LoanAmount'
df.drop( [  260,
       325,  351, 513,535,
        ], inplace = True)

# Outlier’s Index of column 'Loan_Amount_Term'
df.drop([ 75, 109, 168, 179,228, 248, 293, 298, 358, 366, 378, 499,
       515, 572], inplace = True) 


# fetching the number of rows 
rows = df.shape[0]

# displaying the number of rows 
print("Rows: " + str(rows))

#################################################

# fill the null values with mean

df["LoanAmount"]= df["LoanAmount"].fillna(df["LoanAmount"].mean())

df["Loan_Amount_Term"]= df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mean())

#################################################

# X  represents independent columns (Dependants, Married, Education,
#Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History)
# Y represents dependent column (Loan_Status)
from sklearn.model_selection import train_test_split
x,y  = df.iloc[:, 0:-1], df.iloc[:, -1]

# splitting X and y into training and testing sets
x_train , x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train)

# import the regressor
from sklearn.tree import DecisionTreeClassifier

# create a regressor object 
DT_Classifier= DecisionTreeClassifier (criterion="entropy", random_state=0)

# fit the regressor with X and Y data
DT_Classifier.fit(x_train,y_train)

# print the predicted value
DT_Prediction= DT_Classifier.predict(x_test)
print(DT_Prediction)

# calculating accuracy
from sklearn import metrics

print("Decision Tree Accuracy : ", metrics.accuracy_score(DT_Prediction,y_test))

#################################################

from sklearn.model_selection import train_test_split
x,y  = df.iloc[:, 0:-1], df.iloc[:, -1]
x_train , x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
LR_Classifier=LogisticRegression()
LR_Classifier.fit(x_train,y_train)
LR_Prediction=LR_Classifier.predict(x_test)

from sklearn import metrics
print("Logistic Regression Accuracy : ", metrics.accuracy_score(LR_Prediction,y_test))
print("y_predicted",LR_Prediction)
print("y_test",y_test)

#################################################

from sklearn.model_selection import train_test_split
x,y  = df.iloc[:, 0:-1], df.iloc[:, -1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x.shape,x_train.shape,x_test.shape)


from sklearn.svm import SVC
SVM_Classifier=SVC(kernel='linear',random_state=0)

#training svm
SVM_Classifier.fit(x_train,y_train)

# Make the predictions
SVM_Prediction = SVM_Classifier.predict(x_test)
x_train_predict=SVM_Classifier.predict(x_train)#new
 
#accuracy score on training data 
from sklearn import metrics 

#accuracy score on training data 
print("SVM Accuracy in train data:",metrics.accuracy_score( y_train,x_train_predict))

#accuracy score on test data
print("SVM Accuracy in test data:",metrics.accuracy_score(SVM_Prediction,y_test)) 