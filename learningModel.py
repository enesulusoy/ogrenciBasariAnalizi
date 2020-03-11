# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model.logistic import LogisticRegression
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier ,AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from imblearn import over_sampling

#oluşturduğumuz veriyi kullanmak için kütüphane import eder gibi alıyoruz
from student import egitimGirdi,egitimCikti,valGirdi,valCikti

print(egitimGirdi.shape)

#### SENTETİK VERİ ÜRETİMİ
ros= over_sampling.RandomOverSampler()
rosEgitimGirdi, rosEgitimCikti= ros.fit_sample(egitimGirdi,egitimCikti)

print(rosEgitimGirdi.shape)

smote= over_sampling.SMOTE()
smoteEgitimGirdi, smoteEgitimCikti= smote.fit_sample(egitimGirdi,egitimCikti)

print(smoteEgitimGirdi.shape)

ada= over_sampling.ADASYN(ratio='minority')
adasynEgitimGirdi, adasynEgitimCikti= ada.fit_sample(egitimGirdi,egitimCikti)

print(adasynEgitimGirdi.shape)
#print(adasynEgitimGirdi.shape)


#alınan verileri modellerle analiz etme
models= []
models.append(("LR",LogisticRegression()))
models.append(("LDA",LinearDiscriminantAnalysis()))
models.append(("KNN",KNeighborsClassifier()))
models.append(("DCT",DecisionTreeClassifier()))
models.append(("GNB",GaussianNB()))
models.append(("SVC",SVC()))
models.append(("GPC",GaussianProcessClassifier(1.0*RBF(1.0))))
models.append(("MLP",MLPClassifier()))
models.append(("ADB",AdaBoostClassifier()))

print(egitimGirdi)
print(egitimCikti)

#model listesindeki her model için eğitilmiş model oluşturma 
for name,model in models:
     egitimModel= model.fit(egitimGirdi,egitimCikti)
     egitimModelRos= model.fit(rosEgitimGirdi, rosEgitimCikti)
     egitimModelSmote= model.fit(smoteEgitimGirdi, smoteEgitimCikti)
     egitimModelAdasyn= model.fit(adasynEgitimGirdi, adasynEgitimCikti)
     
     egitimSonuc= egitimModel.score(egitimGirdi, egitimCikti)
     egitimSonucRos= egitimModelRos.score(rosEgitimGirdi, rosEgitimCikti)
     egitimSonucSmote= egitimModelSmote.score(smoteEgitimGirdi, smoteEgitimCikti)
     egitimSonucAdasyn= egitimModelAdasyn.score(adasynEgitimGirdi, adasynEgitimCikti)
     
     valSonuc= egitimModel.score(valGirdi, valCikti)
     valSonucRos= egitimModelRos.score(valGirdi, valCikti)
     valSonucSmote= egitimModelSmote.score(valGirdi, valCikti)
     valSonucAdasyn= egitimModelAdasyn.score(valGirdi, valCikti)
     
     
     print(name,":")
     print("ORJ.:",egitimSonuc,"VAL - ORJ.:",valSonuc)
     print("ROS.:",egitimSonucRos,"VAL - ROS.:",valSonucRos)
     print("SMOTE.:",egitimSonucSmote,"VAL - SMOTE.:",valSonucSmote)
     print("ADASYN.:",egitimSonucAdasyn,"VAL - ADASYN.:",valSonucAdasyn)
     print("-------------------------")
     
     
     
from sklearn.utils import class_weight

#class_weight metod dengesiz veri setlerinde kullanılır ve yapılan sınıflandırma da ki oranlara bakarak
#bize buna ait değerleri verir
agirliklar= class_weight.compute_class_weight("balanced", np.unique(egitimCikti),egitimCikti)
print(agirliklar)
     


from sklearn.svm import SVC

a= SVC(random_state=1).fit(egitimGirdi,egitimCikti)
print(a.score(egitimGirdi,egitimCikti))
b= SVC(random_state=1, class_weight={0:1.23,1:0.87,2:0.9}).fit(egitimGirdi,egitimCikti)
print(b.score(egitimGirdi,egitimCikti))


print("-------------")
print("\n")
print("SAMPLE WEIGHT")
a= SVC().fit(smoteEgitimGirdi, smoteEgitimCikti)
print(a.score(smoteEgitimGirdi,smoteEgitimCikti))
print(a.score(valGirdi, valCikti))



a= SVC(kernel="linear").fit(smoteEgitimGirdi,smoteEgitimCikti)
print(a.score(smoteEgitimGirdi,smoteEgitimCikti))

a= SVC(kernel="rbf").fit(smoteEgitimGirdi,smoteEgitimCikti)
print(a.score(smoteEgitimGirdi,smoteEgitimCikti))


a= SVC(kernel="linear", C=0.5).fit(smoteEgitimGirdi,smoteEgitimCikti)
print(a.score(smoteEgitimGirdi,smoteEgitimCikti))

a= SVC(kernel="linear", C=1.0).fit(smoteEgitimGirdi,smoteEgitimCikti)
print(a.score(smoteEgitimGirdi,smoteEgitimCikti))


a= SVC(kernel="linear", C=2.0).fit(smoteEgitimGirdi,smoteEgitimCikti)
print(a.score(smoteEgitimGirdi,smoteEgitimCikti))

a= SVC(kernel="linear", C=20.0).fit(smoteEgitimGirdi,smoteEgitimCikti)
print(a.score(smoteEgitimGirdi,smoteEgitimCikti))

print("---------------------")
print("\n")
print("GRID SEARCH KULLANIMI")

from sklearn.model_selection import GridSearchCV

def svc_param_selection(X,y,nfolds):
     Cler= [0.001,0.01,0.1,0.5,1,2,5,10,100,1000]
     gammalar= [0.001,0.01,0.1,1]
     kernellar= ["linear","rbf"]
     param_grid= {"C":Cler, "gamma":gammalar, "kernel":kernellar}
     
     grid_search= GridSearchCV(SVC(), param_grid, cv=nfolds)
     grid_search.fit(X,y)

     return print(grid_search.best_params_)


svc_param_selection(smoteEgitimGirdi, smoteEgitimCikti, 4)

a= SVC(C=2,kernel="rbf",gamma=1).fit(smoteEgitimGirdi, smoteEgitimCikti)
print(a.score(smoteEgitimGirdi, smoteEgitimCikti))
print(a.score(valGirdi,valCikti))     
     
     
def adb_param_selection(X,y,nfolds):
     n_est= [120,180,300,500]
     lr= [0.001,0.01,0.1,1]
     param_grid= {'n_estimators':n_est, 'learning_rate':lr}
     grid_search= GridSearchCV(AdaBoostClassifier(), param_grid, cv=nfolds)
     grid_search.fit(X,y)
     grid_search.best_params_
     
     return grid_search.best_params_

print(adb_param_selection(valGirdi,valCikti,3))

a= AdaBoostClassifier().fit(smoteEgitimGirdi, smoteEgitimCikti)
print(a.score(smoteEgitimGirdi, smoteEgitimCikti))
print(a.score(valGirdi,valCikti))        
     
     
     
def dtc_param_selection(X,y,nfolds):
     maxdepth= [1.5,8,15,25,30]
     minsampsplit= [1.0,2,5,10,15,100]
     minsampleaf= [1,2,5,10]
     maxfeat= ["log2","sqrt"]
     param_grid= {'maxdepth':maxdepth, 'min_samples_split':minsampsplit, 'min_samples_leaf':minsampleaf, 'max_features':maxfeat}
     grid_search= GridSearchCV(DecisionTreeClassifier(), param_grid, cv=nfolds)
     grid_search.fit(X,y)
     grid_search.best_params_
     
     return grid_search.best_params_

print(dtc_param_selection(valGirdi,valCikti,3))

a= DecisionTreeClassifier().fit(smoteEgitimGirdi, smoteEgitimCikti)
print(a.score(smoteEgitimGirdi, smoteEgitimCikti))
print(a.score(valGirdi,valCikti))       
     
     
     
     
     
