# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
```
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
```
## Algorithm
```
step1:start the program
step2:Import the standard libraries.
step3:Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
step4:Import LabelEncoder and encode the dataset.
step5:Import LogisticRegression from sklearn and apply the model on the dataset.
step6:Predict the values of array.
step7:Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
step8:Apply new unknown values
step9:End
```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ARCHANA S
RegisterNumber:  212223040019
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1.isnull().sum()
data1.duplicated().sum()
data1
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:, : -1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion,display_labels=[True,False])
cm_display.plot()

```
## Output:

![364618252-8e272b59-b8f6-4dd0-98ab-92e31af0b7bd](https://github.com/user-attachments/assets/7629b0b3-3b5a-4288-b2b1-ee59baab36ec)


![364618315-98c954f8-4c2a-43d1-a34f-3acd202edae1](https://github.com/user-attachments/assets/a74e1ff2-9a1e-4455-875f-00404cb99ca4)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
