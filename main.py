from fastapi import FastAPI
app = FastAPI()

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier

df=pd.read_excel("resistant_data_blood.xlsx")

X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=10)

#X = np.array(ct.fit_transform(X), dtype=np.float64)

sc=MaxAbsScaler()
X_train_new=sc.fit_transform(X_train)
X_test_new=sc.transform(X_test)
#X_train_new = np.array(sc.fit_transform(X_train), dtype=np.float64)
#X_test_new = np.array(sc.fit_transform(X_test), dtype=np.float64)

model=KNeighborsClassifier(n_neighbors=4,p=1)
model=model.fit(X_train_new,y_train)

#s=model.predict([sapmle])'''

@app.get("/my-first-api")
def infection_detection(age = None, gender= None, fever = None, Bone_merrow_transplantation = None, hb = None, platet = None,  crp = None, procalictonin = None, e_colli = None, klebsilla = None, pseudomonas = None):
    result = ""
    Result1 = 0
    Result2 = 0
    Result3 = 0
    print(age, hb, fever, crp)
    if (age == None or gender == None or fever == None or Bone_merrow_transplantation == None or hb == None or platet == None or crp == None or procalictonin == None or e_colli == None or klebsilla == None or pseudomonas == None):
        result = 'Someting missing, please provide all variable data '

    else:
        if float(e_colli)<= -10:
            Result1 = 1
        if float(klebsilla)<= -10:
            Result2 = 1
        if float(pseudomonas)<= -10:
            Result3 = 1
        age = np.float64(age)
        Result1 = np.float64(Result1)
        Result2 = np.float64(Result2)
        Result3 = np.float64(Result3)
        gender = np.float64(gender)
        fever = np.float64(fever)
        print(type(fever))
        Bone_merrow_transplantation = np.float64(Bone_merrow_transplantation)
        hb = np.float64(hb)
        crp = np.float64(crp)
        procalictonin = np.float64(procalictonin)
        e_colli = np.float64(e_colli)
        klebsilla = np.float64(klebsilla)
        pseudomonas = np.float64(pseudomonas)
        sapmle=[age, gender, fever, Bone_merrow_transplantation, hb, platet, crp, procalictonin, e_colli, Result1, klebsilla, Result2, pseudomonas, Result3]
        result=model.predict([sapmle])
        print (result)
    

    return "hello"