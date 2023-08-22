"""
Created on Wed Aug 16 19:35:51 2023
"""
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import  AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.feature_selection import mutual_info_classif


#Data import and Feature engineering  

#========================================================================================================
bank_case = pd.ExcelFile('Banking Case - Data.xlsx')
tran_data = pd.read_excel(bank_case, 'Transaction Data')
cust_demog = pd.read_excel(bank_case, 'Customer Demographics')

#Balance
tran_data['balance'].replace(['??', '????','???','//??','??//'], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(tran_data[['balance']])
tran_data['balance'] = imputer.transform(tran_data[['balance']]).astype(int)

#loan
tran_data['loan'] = tran_data['loan'].fillna('no')

#Contact
tran_data['contact'].replace(['Mobile', 'Tel','?'], ['cellular', 'telephone', np.nan], inplace=True)
tran_data['contact'] = tran_data['contact'].fillna('unknown')

#Duration
tran_data['duration'] = tran_data['duration'].abs()
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(tran_data[['duration']])
tran_data['duration'] = imputer.transform(tran_data[['duration']]).astype(int)

#Last contact day
tran_data['last_contact_day'].replace([-2,-5,-7,-9], [2,5,7,9], inplace=True)

#poutcome
tran_data['poutcome'].replace(['?', '????','unknown','other'], np.nan, inplace=True)

#Count Txn
tran_data['Count_Txn'] = tran_data['Count_Txn'].abs()
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(tran_data[['Count_Txn']])
tran_data['Count_Txn'] = imputer.transform(tran_data[['Count_Txn']]).astype(int)

# customer demographic dataset
# Job
cust_demog['job'].replace(['blue collar','admin.'], ['blue-collar','admin'], inplace=True)
cust_demog['job'] = cust_demog['job'].fillna('unknown')

#Marital
cust_demog['marital'] = cust_demog['marital'].fillna('married')

#Education
cust_demog['education'].replace(['ter tiary','Primary'], ['tertiary','primary'], inplace=True)
cust_demog['education'] = cust_demog['education'].fillna('unknown')

#Annual Income
cust_demog['Annual Income'].replace(['\\'], np.nan, inplace=True)
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer = imputer.fit(cust_demog[['Annual Income']])
cust_demog['Annual Income'] = imputer.transform(cust_demog[['Annual Income']]).astype(int)

#Merge the two datasets Transaction data and customer demographic
dataset = pd.merge(tran_data, cust_demog, on='Customer_number')



#Dropping customers of age greater than 100 and droping poutcome due to lots of missing dataset and rows with no target values
indexAge = dataset[ (dataset['age'] > 100)  ].index
dataset.drop(indexAge , inplace=True)
dataset = dataset[dataset['Term Deposit'].notna()]
dataset = dataset.drop(columns=['Sno','poutcome','Customer_number'])

#Annual income distribution of customers with loan and non-loan
plt.figure(1)
dataset[dataset['loan'] == 'yes']['Annual Income'].plot(kind='kde')
dataset[dataset['loan'] == 'no']['Annual Income'].plot(kind='kde')
plt.legend()

#Plotting bar graph of the average balance of each profession with loan customers amd non-loan customers. 
ddi = pd.DataFrame({'No_of_trx' : dataset.groupby( [ "loan", "job"] )["balance"].nunique()}).reset_index()
ddq = pd.DataFrame({'quantity' : dataset.groupby( [ "loan", "job"] )["balance"].sum()}).reset_index()
dddt1 = pd.merge(ddq, ddi, on= ['loan' , 'job'])
dddt1["avg_balance"] = dddt1["quantity"]/dddt1["No_of_trx"]

ddi1 = pd.DataFrame({'No_of_trx' : dataset.groupby( [ "loan", "job"] )["Annual Income"].nunique()}).reset_index()
ddq1 = pd.DataFrame({'quantity' : dataset.groupby( [ "loan", "job"] )["Annual Income"].sum()}).reset_index()
dddt2 = pd.merge(ddq1, ddi1, on= ['loan' , 'job'])
dddt2["avg_income"] = dddt2["quantity"]/dddt2["No_of_trx"]

plt.figure(2)
ax = dddt1[dddt1['loan'] == 'no'].plot.bar(x='job', y='avg_balance', rot=0)
plt.ylim(0, 6000)
ax = dddt1[dddt1['loan'] == 'yes'].plot.bar(x='job', y='avg_balance', rot=0)
plt.legend()
plt.ylim(0, 6000)

#Encoding categorical values
categorical_features = [
        "Insurance",
        "housing",
        "loan",
        "contact",
        "Term Deposit",
        "job",
        "marital",
        "education",
        "Gender",
    ]

encoder = OrdinalEncoder()
encoder.fit(dataset[categorical_features])
dataset[categorical_features] = encoder.transform(dataset[categorical_features])
y = dataset.iloc[:, 9]
X = dataset.drop(columns=['Term Deposit'])

#Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

#==============================================================================================================

#Feature Selection using mutual importance
mi = mutual_info_classif(X_train, y_train)
mi_series = pd.Series(mi)
mi_series.index = X_train.columns
mi_series = mi_series.sort_values(ascending=False)

#Training the dataset and predicting the test dataset
classifier = AdaBoostClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy = ", acc)
yreport_metrics = classification_report(y_test, y_pred, digits=3,  output_dict=True)
acm = confusion_matrix(y_test, y_pred)
