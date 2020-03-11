# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#kullanıcağımız veriyi çekme işlemi ve ";" göre ayırmasını anlatma
veri= pd.read_csv("student/student-mat.csv",delimiter=";")


#çektiğimiz verinin gerek duymadığımız sutunları atma işlemi
veri= veri.drop(["school","sex","age","address","Mjob","Fjob","reason","guardian","activities","nursery","higher","internet","romantic","famrel","freetime","goout","Dalc","Walc","health","absences"],axis=1)


#sözel değerleri sayisal değerlere çevirme(scala oranını 0-1 arasında tutmak için)
veri= veri.replace({"GT3":1,"LE3":0,"yes":1,"no":0,"T":1,"A":0})

sinavsonucu= veri[["G1","G2","G3"]]


#veriyi normalize etmek için kullanılan kütüphaneden nesne oluşturma
min_max_scaler= MinMaxScaler()


#veriyi normalize etme işlemi
veri= min_max_scaler.fit_transform(veri)


#normalize olan verinin tipi değişmiş olma itimaline karşı dataframe çevirme işlemi
veri= pd.DataFrame(veri)


#normalizasyonu kendimiz yapabilmek için formülü oluşturduk sebebi verilerde max veri
#20 olması gerekirken 19 olarak görülmüştür o da buna göre almıştır bizde normalde
#olması gereken veriye göre alması için kendimiz oluşturduk
sinavsonucunormalize= (sinavsonucu-0)/(20-0)


trainig, kalanveri= train_test_split(veri, test_size=0.2)
validation, test= train_test_split(kalanveri, test_size=0.5)

#sınav sutunlarını dışarda normalize ettik o yuzden veriden kaldırıp normalize halini
#veriye ekleyeceğiz (nedeni ise veri içerisindeyken son nottan eksik veya küçük nottan
#fazla puana sahip olan bölümler mevcut bunun önüne geçmek için normalize işlemini
#elle yaptık ve formülü yazdık sonra da bunu veri setine ekleyeceğiz)
veri= veri.drop(veri.columns[[10,11,12]],axis=1)

normalizeveri= veri.join(sinavsonucunormalize)

#3 notu toplayıp 0-1 yerine 0-3 arasında bir normalizasyon yapıp veriye ekleyelim
sinavsonuctoplam= sinavsonucunormalize[["G1","G2","G3"]].sum(axis=1)


#toplam sonuclari ekleyeceğimiz için g1,g2,g3 ü kaldıralım
normalizeveri= normalizeveri.drop(normalizeveri[["G1","G2","G3"]],axis=1)
#drop(normalizeveri.columns[[10,11,12]],axis=1) bu da kullanılabilir 

#şimdi toplam notlari ekleme işlemini gerçekleştirelim
normalizeveri["toplam"]= sinavsonuctoplam



#toplam sonucları plot ile çizdirelim
#import matplotlib.pyplot as plt

#çizdireceğimiz grafik türünü seçtik
#plt.hist(sinavsonuctoplam)
#plt.show()


#0-1.3 , 1.3-1.8 , 1.8-3 şeklinde sınıflandırma işlemi yapalım (3 lü sınıflandırma)
#  0        1         2   değerleri vereceğiz

birinci= normalizeveri.toplam<1.3
ikinci= (normalizeveri.toplam>=1.3) & (normalizeveri.toplam<=1.8)
ucuncu= normalizeveri.toplam>=1.8


#loc[] ile ilk değerde verilen değerleri ikinci değerdeki sutuna göre yapılandırır
normalizeveri.loc[birinci,"toplam"]= 0
normalizeveri.loc[ikinci,"toplam"]= 1
normalizeveri.loc[ucuncu,"toplam"]= 2

#elimizde bulunan verinin %80 i eğitim verisi olarak %20 si onaylama olarak 
#olarak kullanılmasi için train_test_split kullanıyoruz
egitim, validation= train_test_split(normalizeveri, test_size=0.2, random_state=7)


#verileri deneme ve sonuç olarak ayırıyoruz
egitimGirdi= egitim.drop(["toplam"],axis=1)
egitimCikti= egitim.toplam


valGirdi= validation.drop(["toplam"],axis=1)
valCikti= validation.toplam



"""
#normalize verinin çizdirilmesi
import matplotlib.pyplot as plt

#çizdireceğimiz grafik türünü seçtik
plt.hist(normalizeveri.toplam)
plt.show()
"""




 

