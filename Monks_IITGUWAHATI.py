import pandas as pd 
import numpy as np
!pip install -q keras
import keras
data=pd.read_csv('Training_dataset_Original.csv')
test=pd.read_csv('Leaderboard_dataset.csv')
y=data.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
data['mvar47'] = labelencoder_X.fit_transform(data['mvar47'])
test['mvar47'] = labelencoder_X.fit_transform(test['mvar47'])
data=data.drop(['application_key','mvar11','mvar6','mvar17','mvar23','mvar30','mvar31','mvar40','mvar41','mvar45','mvar46','default_ind'],axis=1)
test=test.drop(['application_key','mvar11','mvar6','mvar17','mvar23','mvar30','mvar31','mvar40','mvar41','mvar45','mvar46'],axis=1)
data=data.apply(pd.to_numeric,args=('coerce',))
test=test.apply(pd.to_numeric,args=('coerce',))
data=data.fillna(data.mean())
test=test.fillna(test.mean())
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
data=scaler.fit_transform(data)
test=scaler.transform(test)
val=pd.read_csv('Evaluation_dataset.csv')
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
val['mvar47'] = labelencoder_X.fit_transform(val['mvar47'])
val=val.drop(['application_key','mvar11','mvar6','mvar17','mvar23','mvar30','mvar31','mvar40','mvar41','mvar45','mvar46'],axis=1)
val=val.apply(pd.to_numeric,args=('coerce',))
val=val.fillna(val.mean())
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
val=scaler.fit_transform(val)
import keras
from keras.models import Sequential
from keras import regularizers
from keras.layers import Dense,Dropout
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 74, init = 'uniform', activation = 'relu', input_dim = 37))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 74, init = 'uniform',kernel_regularizer=regularizers.l2(0.01), activation = 'relu'))
 #classifier.add(Dropout(.8, noise_shape=None, seed=None))
classifier.add(Dense(output_dim = 55, init = 'uniform', activation = 'tanh'))
classifier.add(Dropout(.7, noise_shape=None, seed=None))
classifier.add(Dense(output_dim = 37,init = 'uniform', activation = 'tanh'))
classifier.add(Dropout(.7, noise_shape=None, seed=None))
# Adding the second hidden layer
# Adding the output layer
classifier.add(Dense(output_dim = 19, init = 'uniform',kernel_regularizer=regularizers.l2(0.01), activation = 'sigmoid'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(data, y, batch_size = 50, nb_epoch = 80)
ypredtnn=classifier.predict(test)
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0)
classifier1.fit(data, y)
y_pred = classifier1.predict(val)

# Making the Confusion Matrix
from sklearn.metrics import accuracy_score
print ("Train Accuracy :: ", accuracy_score(y, classifier1.predict(data)))
y_test=classifier1.predict(test)
from sklearn.metrics import zero_one_loss
print ("Train loss :: ", float(zero_one_loss(y, classifier1.predict(data))))
for i in range(0,len(y_test)):
    if(y_test[i]==1 and ypredtnn[i]<.35 ):
       y_test[i]=0
    elif(y_test[i]==0 and ypredtnn[i]>.58):
      y_test[i]=1
tet=pd.read_csv('Evaluation_dataset.csv')
a=tet['application_key']
df = pd.DataFrame({"application_key" : a, "default_ind" : y_test})
df.to_csv("valMonks_IITGUWAHATI_16.csv", index=False)
dat=pd.read_csv('vvv.csv')
cl2=dat['13']
cl1=dat['1']
for i in range(0,len(cl1)):
    if(cl2[i]>=.5):
        cl2[i]=1
    else:
        cl2[i]=0
