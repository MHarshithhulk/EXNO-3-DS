## EXNO-3-DS
# Name : M Harshith
# Reg no : 212224040206

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

<img width="482" height="548" alt="image" src="https://github.com/user-attachments/assets/4989933e-ea19-47e3-b37e-3df114e356b6" />

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])

<img width="777" height="355" alt="image" src="https://github.com/user-attachments/assets/96c8ff63-2eb5-4860-9644-21810e843442" />

df['bo2']=e1.fit_transform(df[["ord_2"]])
df

<img width="580" height="527" alt="image" src="https://github.com/user-attachments/assets/5c8efdca-b538-414d-8fbc-4171d3b97eb6" />

le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

<img width="657" height="571" alt="image" src="https://github.com/user-attachments/assets/cfdc7593-5cbe-4534-82fb-96771a05d30b" />

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))

df2=pd.concat([df2,enc],axis=1)
df2

<img width="1065" height="522" alt="image" src="https://github.com/user-attachments/assets/2396a260-736a-4c41-8853-ea028606e8e8" />

pd.get_dummies(df2,columns=["nom_0"])

<img width="1282" height="502" alt="image" src="https://github.com/user-attachments/assets/7ca0c06e-7222-4851-b011-b7c32e1723be" />

pip install --upgrade category_encoders

from category_encoders import BinaryEncoder
df=pd.read_csv("data.csv")
df

<img width="717" height="547" alt="Screenshot 2025-09-30 110300" src="https://github.com/user-attachments/assets/60b605d0-e365-4846-bdfb-e95ccc29c336" />

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df

<img width="716" height="547" alt="image" src="https://github.com/user-attachments/assets/a5800096-0792-4673-b349-3bfcf036fe9c" />

dfb=pd.concat([df,nd],axis=1)
dfb

<img width="1028" height="515" alt="image" src="https://github.com/user-attachments/assets/d3795bd3-5ac6-42ac-8ba3-7a80d3b77c05" />

from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC

<img width="858" height="622" alt="image" src="https://github.com/user-attachments/assets/e8d143c4-da35-44a8-8e34-9af073c01bb8" />

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df

<img width="1157" height="661" alt="image" src="https://github.com/user-attachments/assets/1ce43a41-c957-4a42-b32a-cbb6cc848e57" />

df.skew()

<img width="458" height="296" alt="image" src="https://github.com/user-attachments/assets/52bf2847-3f7b-4c8a-b062-ef77fbef36a9" />

np.log(df["Highly Positive Skew"])

<img width="471" height="617" alt="image" src="https://github.com/user-attachments/assets/7653a469-61d4-43ad-8e0a-b0fb249a2e35" />

np.reciprocal(df["Moderate Positive Skew"])

<img width="547" height="612" alt="image" src="https://github.com/user-attachments/assets/f0d276d2-6f49-4588-8c72-9f86cb604e77" />

np.sqrt(df["Highly Positive Skew"])

<img width="480" height="621" alt="image" src="https://github.com/user-attachments/assets/3792c86e-5638-4437-a166-66b60d708886" />

np.square(df["Highly Positive Skew"])

<img width="633" height="618" alt="image" src="https://github.com/user-attachments/assets/6f0002bd-e27b-4eaf-8dca-fbf420ea4f03" />

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df

<img width="1451" height="597" alt="image" src="https://github.com/user-attachments/assets/550c324d-f503-42a5-9f45-4cb196ca487c" />

df.skew()

<img width="601" height="347" alt="image" src="https://github.com/user-attachments/assets/4442ff65-fe87-4508-b2b6-cc32e8c4fc01" />

df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()

<img width="1030" height="725" alt="image" src="https://github.com/user-attachments/assets/1648b294-7eee-442c-b22f-7524be32d31d" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df

<img width="1743" height="662" alt="image" src="https://github.com/user-attachments/assets/b49de518-c5ab-4132-a52f-85f4ecc95d4c" />

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

<img width="866" height="708" alt="image" src="https://github.com/user-attachments/assets/3ec0781c-1639-4a1e-8ea4-c5764cc370a1" />

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()

<img width="837" height="633" alt="image" src="https://github.com/user-attachments/assets/25f5c41a-ece1-4c59-b204-1f40c8286f94" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()

<img width="830" height="632" alt="image" src="https://github.com/user-attachments/assets/cc8e111d-414e-4d94-b402-27358bad2228" />

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()

<img width="933" height="663" alt="image" src="https://github.com/user-attachments/assets/2494e94f-b971-49e9-90ce-04db2044b253" />

dt=pd.read_csv("titanic_dataset.csv")
dt

<img width="1485" height="592" alt="image" src="https://github.com/user-attachments/assets/149362a1-828b-4f7d-a9fa-9b04a3c79b9c" />

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45')
plt.show()

<img width="903" height="706" alt="image" src="https://github.com/user-attachments/assets/d3d7e6fe-41f1-4e42-9b78-8f923d7bf823" />

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()

<img width="793" height="633" alt="image" src="https://github.com/user-attachments/assets/b36bb58e-7c33-4c71-b12f-7c2eb78e7957" />

       
# RESULT:   
Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.
       
