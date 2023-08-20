import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st
df=pd.read_excel("resistant_data_blood.xlsx")


X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=110)

sc=MaxAbsScaler()
X_train_new=sc.fit_transform(X_train)
X_test_new=sc.transform(X_test)

model=KNeighborsClassifier(n_neighbors=5,p=1)
model=model.fit(X_train_new,y_train)

print("Training accuracy: ",model.score(X_train_new,y_train))
print("Testing accuracy : ",model.score(X_test_new,y_test))


st.title('Resistant')
st.title(':blue[Blood test]:')


Age = st.number_input("Age")
options = ["Male", "Female"]
selectbox_selection = st.selectbox("Select Gender", options)
#st.write(f"Gender selected is {selectbox_selection}")
Fever = st.number_input("Fever")
options1 = ["Yes", "No"]
selectbox_selection = st.selectbox("Bone_merrow_transplantation", options1)
HB = st.number_input("HB")
platet = st.number_input("platet")
CRP= st.number_input("CRP")
Procalictonin =st.number_input("Procalictonin")
E_colli= st.number_input("CTX-M")
Result1 =0
Klebsilla = st.number_input("KPC")
Result2 = 0
Pseudomonas= st.number_input("NDM")
Result3 = 0
submit=st.button("Result")
gender = 1
Bone_merrow_transplantation=1

if float(E_colli)<= -10:
    Result1 = 1
if float(Klebsilla)<= -10:
    Result2 = 1
if float(Pseudomonas)<= -10:
    Result3 = 1
if selectbox_selection == "FEMALE":
    gender = 0
if selectbox_selection == "NO":
    Bone_merrow_transplantation=0


sapmle=[Age, gender, Fever, Bone_merrow_transplantation, HB, platet, CRP, Procalictonin, E_colli, Result1, Klebsilla, Result2, Pseudomonas, Result3]
s=model.predict([sapmle])
st.write(s)
print(s)