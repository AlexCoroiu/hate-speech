from sklearn.metrics import f1_score, confusion_matrix, recall_score, precision_score
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from prettytable import PrettyTable
import pickle
import numpy
import processing
import os


train_data_np = numpy.array(processing.train_data)
train_labels_np = numpy.array(processing.train_labels)
print(train_labels_np)

test_data_np = numpy.array(processing.test_data)
test_labels_np = numpy.array(processing.test_labels)
print(test_labels_np)

classifier = SVC(kernel='linear', class_weight='balanced')
parameters = {'clf__C':[0.01,0.1,1,10,100]}
#classifier.fit(train_data_np,train_labels_np)


svc_pipe = Pipeline(steps = [('vect', processing.vectorizer),
                             ('clf', classifier)])

best_model = GridSearchCV(svc_pipe, parameters, cv=10, scoring ='f1')
best_model.fit(train_data_np, train_labels_np)

#print rezults
means = best_model.cv_results_['mean_test_score']
stds = best_model.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, best_model.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


print(best_model.best_params_)
print('f1 train = ', best_model.best_score_)

#cross_validation
#scores = cross_val_score(svc_pipe, train_data_np, train_labels_np, scoring='f1', cv = 10)
#print('f1 train = ', scores.mean())
#f1 linear = 0.7802570829581987

#train
#svc_pipe.fit(train_data_np, train_labels_np)

#test
predicted = best_model.predict(test_data_np)

#metrics
tn,fp,fn,tp = confusion_matrix(test_labels_np, predicted).ravel()
matrix = PrettyTable(['TN','FP','FN','TP'])
matrix.add_row([tn,fp,fn,tp])
print(matrix)
print('recall = ',recall_score(test_labels_np,predicted))
print('precision = ',precision_score(test_labels_np,predicted))
print('f1 test = ',f1_score(test_labels_np,predicted))
#f1 linear = 0.3374265883609183

#save model
file = 'SVC.txt'

if os.path.exists(file):
    os.remove(file)

with open(file,'w+b') as clf:
    pickle.dump(best_model, clf)

#degree, gamma, coef0

