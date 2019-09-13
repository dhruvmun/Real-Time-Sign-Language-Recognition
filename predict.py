import numpy as np 
from keras.models import load_model
my_model = load_model('sign_language_model2.h5')

# Dictionary to convert numerical classes to alphabet
idx2alph = {0:'a',1:'b',2:'c',3:'d',4:'e',5:'f',6:'g',7:'h',8:'i',9:'j',10:'k',11:'l',12:'m',13:'n',14:'o',
            15:'p',16:'q',17:'r',18:'s',19:'t',20:'u',21:'v',22:'w',23:'x',24:'y',25:'z'}

def predict(img):
	img = img.reshape(1, 28, 28, 1)
	y_pred = my_model.predict(img, batch_size=1, verbose=0)
	score = max(y_pred)
	y_pred = np.argmax(y_pred, axis=1)
	return y_pred, score


import pandas as pd 
df = pd.read_csv('sign_mnist_test.csv')

x = df.iloc[:,1:].values.astype('float32')
y = df.iloc[:,0].values.astype('int32')
print(y[:10])

x = x.reshape(x.shape[0], 28, 28, 1)

pred, score = predict(x[1])

print(pred)
#print(score)
a = pred[0]
print(idx2alph[a])