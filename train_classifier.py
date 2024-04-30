import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data_dict = pickle.load(open('./data.pickle','rb')) # dict
import numpy as np

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=0.2,shuffle=True,stratify=labels)
# split the dataset but same proportion of labels in train and test - stratify
# xtest and ytest have same proportion of all 3 diff labels
# 1/3 a, 1/3 b, 1/3 c

model = RandomForestClassifier()
model.fit(x_train,y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict,y_test)
print(f"{score*100}% of samples were correct")
# 100% accuracy 

f = open('model.p','wb')
pickle.dump({'model':model},f)
f.close()
