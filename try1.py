from keras.models import model_from_json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
import operator
from numpy import random

#y_binary = to_categorical(y_int)
dataset = pd.read_excel("Contact_Model.xlsx")

x=dataset.iloc[0:500,0:66].values
y=dataset.iloc[0:500,66].values
labelEncoder_col= LabelEncoder()
x[:,0] = labelEncoder_col.fit_transform(x[:,0])
x[:,1] = labelEncoder_col.fit_transform(x[:,1])
x[:,2]/=100
x[:,3] = labelEncoder_col.fit_transform(x[:,3])
x[:,4] = labelEncoder_col.fit_transform(x[:,4])
x[:,5] = labelEncoder_col.fit_transform(x[:,5])
y[:] = labelEncoder_col.fit_transform(y[:])
for i in range(6,66):
    x[:,i] = labelEncoder_col.fit_transform(x[:,i])

'''
print("X_0_:{}".format(x[:,0]))
print("X_1_:{}".format(x[:,1]))
print("X_3_:{}".format(x[:,3]))
print("X_4_:{}".format(x[:,4]))
print("X_5_:{}".format(x[:,7]))
print("Y_0_:{}".format(y[:]))
'''

#X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=.1, random_state=0 )
#Y_train = to_categorical(Y_train, 3)
y_train = to_categorical(y, 3)

model = Sequential()
model.add(Dense(66, input_dim=66, activation='sigmoid'))
model.add(Dense(66, activation='sigmoid'))
model.add(Dense(66, activation='relu'))
model.add(Dense(66, activation='sigmoid'))


model.add(Dense(3, activation='softmax'))


#compile model
model.compile(loss='MSE', optimizer=SGD(), metrics=['accuracy'] )

model.summary()

#model.fit(X_train, Y_train, epochs=500 , batch_size=30)

model.fit(x, y_train, epochs=600 , batch_size=25)



Y_pred = model.predict(x)
#Y_pred.shape
#print(Y_pred)
Y_pre=[]
for i in Y_pred:
    max_index, max_value = max(enumerate(i), key=operator.itemgetter(1))
    Y_pre.append(max_index)


q=np.asarray(Y_pre,dtype=np.int)
y=np.asarray(y,dtype=np.int)

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y, q)
true=0
false=0
for i in range(0,500):
    if(Y_pre[i]==y[i]):
        true+=1
    else:
        false+=1

print(confusion)    
'''                                             #save model   
fname = "weights_contact_model_sam_88_acc.hdf5"
model.save_weights(fname,overwrite=True)
'''
                                                #load model
'''
fname = "weights_contact_model_sam.hdf5"
model.load_weights(fname)
'''
