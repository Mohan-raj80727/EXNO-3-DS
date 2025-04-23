## EXNO-3-DS

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
~~~
import pandas as pd
df=pd.read_csv("/content/Encoding Data (1).csv")
df
~~~
![image](https://github.com/user-attachments/assets/228cf0f6-a67a-401a-a31c-a632546101f7)
~~~
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
~~~
![image](https://github.com/user-attachments/assets/31068673-aa16-4582-9ecd-9ab3fceb560c)
~~~
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
~~~
![image](https://github.com/user-attachments/assets/7c478a74-3c68-45c7-be46-ec0aebd21ba3)
~~~
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
~~~
![image](https://github.com/user-attachments/assets/20c8fa95-4dc6-44cb-b8b8-a2b30520c236)
~~~
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False) # Change 'sparse' to 'sparse_output'
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
~~~
![image](https://github.com/user-attachments/assets/faa24f53-5c3c-4941-b54f-9f9703b8820b)
~~~
pd.get_dummies(df2,columns=["nom_0"])
~~~
![image](https://github.com/user-attachments/assets/fc29521d-a7b4-4c9e-93f6-2a5afac89b49)
~~~
!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder
import pandas as pd
df=pd.read_csv("/content/data (1).csv")
df

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df

dfb=pd.concat([df,nd],axis=1)
dfb
~~~
![image](https://github.com/user-attachments/assets/b86b8782-6ba3-41cf-be9a-804c320f2f29)
~~~
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
~~~
![image](https://github.com/user-attachments/assets/06a000ef-a4b3-4ed5-a6bc-f973012d610e)
~~~
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform (1).csv")
df
~~~
![image](https://github.com/user-attachments/assets/74c1e361-cb95-4435-bcbf-0261cac7a819)
~~~
df.skew()
~~~
![image](https://github.com/user-attachments/assets/2a88208f-5d1a-403d-983f-405ee5abacc3)
~~~
np.log(df["Highly Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/d2ba4e22-cffe-4537-8dfe-d2401c30ecc7)
~~~
 np.reciprocal(df["Moderate Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/b33e34ad-ad05-47d8-8d0a-c81be3b1fe5e)
~~~
np.sqrt(df["Highly Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/ad0f565d-babc-40af-9cc4-5884fcee4fe8)
~~~
np.square(df["Highly Positive Skew"])
~~~
![image](https://github.com/user-attachments/assets/e2e36114-b96d-42fb-ae6f-6d683c484304)
~~~
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
~~~
![image](https://github.com/user-attachments/assets/a684ad95-7455-4845-91dc-c2ab5497686c)
~~~
df.skew()
~~~
![image](https://github.com/user-attachments/assets/41f12330-a895-4190-a56b-1ae9a12133e7)
~~~
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
~~~
![image](https://github.com/user-attachments/assets/faa3c789-e5e8-4240-aece-203ff218f00e)
~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
~~~
![image](https://github.com/user-attachments/assets/795843fa-e99e-440d-9156-5b0044f2001b)
~~~
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/4763f159-6489-4a4c-ae45-eac674a269d7)
~~~
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/0497b1bd-7e8d-4993-a8a8-3a71bbd49361)
~~~
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/3a8a49bb-9076-43f5-9601-4606efc5e1e5)
~~~
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/3f196d1a-8fe5-441f-9e24-9791afde2b12)
~~~
dt=pd.read_csv("/titanic_dataset (1).csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
~~~
![image](https://github.com/user-attachments/assets/18c50f9c-1076-4f68-8ab8-ea0c52090901)
~~~
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
~~~
![image](https://github.com/user-attachments/assets/1ecad441-22bc-4a74-a8de-647f3a06805f)

# RESULT:

Thus the given data, Feature Encoding, Transformation process and save the data to a file
was performed successfully.

       
