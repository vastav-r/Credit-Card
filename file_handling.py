from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.cross_validation import train_test_split
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
import operator
from sklearn.metrics import confusion_matrix


model = load_model('my_model.h5')       #complete model
'''
model = Sequential()
model.add(Dense(66, input_dim=66, activation='relu'))
model.add(Dense(66, activation='relu'))
model.add(Dense(3, activation='softmax'))


#compile model
model.compile(loss='MSE', optimizer=SGD(), metrics=['accuracy'] )

model.summary()

fname = "weights_contact_model_sam.hdf5"
model.load_weights(fname)
'''
dataset = pd.read_excel("Contact_Model.xlsx")
x_name=dataset.iloc[0:500,0:66]
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
    
    
Y_pred = model.predict(x)
Y = to_categorical(y, 3)
#Y_pred.shape
#print(Y_pred)
Y_pre=[]
for i in Y_pred:
    max_index, max_value = max(enumerate(i), key=operator.itemgetter(1))
    Y_pre.append(max_index)    
#print (Y_pre)
true=0
false=0
for i in range(0,500):
    if(Y_pre[i]==y[i]):
        true+=1
    else:
        false+=1

accuracy=true/500
print(accuracy)
q=np.asarray(Y_pre,dtype=np.int)
Y_test=np.asarray(y,dtype=np.int)
confusion=confusion_matrix(Y_test, q)        
print(confusion)
def turn(a):
    a=int(a)
    if(a==0):
        return 'Email'
    elif(a==1):
        return 'Phone'
    elif(a==2):
        return 'Sms'
    else:
        return ''
file=open(r'write_data1.csv','w')
for i in range(0,500):
    file.write("{},{},{},{},{},{},{}\n".format(x_name.iloc[i][0],x_name.iloc[i][1],x_name.iloc[i][2],x_name.iloc[i][3],x_name.iloc[i][5],turn(Y_pre[i]),x_name.iloc[i][4]))
file.close()


'''
model.save('my_model.h5') 
'''
