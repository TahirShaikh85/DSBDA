import pandas as pd

data = pd.read_csv(r'Social_Network_Ads.csv')
print("dataset -> \n",data)

print('Gender column -> \n', data['Gender'])

print('isnull -> \n', data.isnull().sum())

print('datatypes -> \n', data.dtypes)

# converts the categorical values 'Male' and 'Female' in the 'Gender' column into numerical values
data['Gender'] = data['Gender'].map({'Male':1,'Female':0})
print('numercial values of Gender col -> \n',data['Gender'])


x = data.drop(['Age'],axis = 1)
y = data['Age']



from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)


from sklearn.preprocessing import StandardScaler

st_x = StandardScaler()
xtrain = st_x.fit_transform(xtrain)
xtest = st_x.transform(xtest)


from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)
classifier.fit(xtrain, ytrain)

y_pred = classifier.predict(xtest)
print("y_pred --> \n",y_pred)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
print("confusion matrix --> \n", cm)
