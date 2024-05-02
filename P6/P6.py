import pandas as pd

data = pd.read_csv(r"Social_Network_Ads.csv")

print('Gender column -> \n', data['Gender'])

print('isnull -> \n', data.isnull().sum())

data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
print('Gender column -> \n', data['Gender'])

print("Datatypes: --> \n", data.dtypes)

# x contains all the columns from data except the 'EstimatedSalary' column
x = data.drop(['EstimatedSalary'],axis=1)
y = data['EstimatedSalary']


from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
    
st_x = StandardScaler() 
   
xtrain = st_x.fit_transform(xtrain)  
  
xtest = st_x.transform(xtest)  


###################### Naive  Bayes ###########

from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()

gaussian.fit(xtrain, ytrain)

y_pred = gaussian.predict(xtest)


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(ytest, y_pred)

print("Accuracy is: \n", accuracy)


from sklearn.metrics import precision_score, recall_score

precision  = precision_score(ytest, y_pred, average ='micro')

print("Precision is: \n", precision)

recall = recall_score(ytest, y_pred, average='micro')

print("Recall is: \n", recall)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(ytest, y_pred)

print("Confusion matrix is: \n", cm)
