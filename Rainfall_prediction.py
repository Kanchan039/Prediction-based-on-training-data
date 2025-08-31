import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Load datasets
train_df=pd.read_csv(r"C:\Users\Lenvo\Downloads\Rain_Train.csv")
test_df=pd.read_csv(r"C:\Users\Lenvo\Downloads\Rain_Test.csv")

#Selecting features and target variable 
features=['day','pressure','maxtemp','temparature','mintemp','dewpoint','humidity','cloud','sunshine','winddirection','windspeed']
target='rainfall'

X=train_df[features]
y=train_df[target]

#Splitting into training and validation sets
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=42)

#Training the Random Forest Model
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)

#Predicting on validation set
y_pred=model.predict(X_val)
accuracy=accuracy_score(y_val,y_pred)
print(f"Validation Accuracy:{accuracy*100:.2f}%")

#Handling missing values in test data
test_df.fillna(test_df.median(),inplace=True)

#Predicting rainfall for the test dataset
test_df['rainfall']=model.predict(test_df[features])

#Saving the results
test_df.to_csv("Rain_Test_Predictions.csv",index=False)
print("Predictions saved to Rain_Test_Predictions.csv")