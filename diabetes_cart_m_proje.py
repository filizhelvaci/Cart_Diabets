import numpy as np
import pandas as pd
import pydotplus
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score

from helpers.data_prep import *
from helpers.eda import *
from sklearn.model_selection import *
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.width', 170)

df_a=pd.read_csv("dataset/diabetes.csv")
"""
Pregnancies = Hamile kalma sayısı
Glucose = Glikoz
Blood Pressure = Kan basıncı
Skin Thickness = Deri kalınlığı
Insulin = İnsülin
BMI (Body Mass Index) = Beden kitle endeksi
Diabetes Pedigree Function = Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon
Age = Yaş
Outcome = Diyabet olup olmadığı bilgisi (bu bizim target yani hedefimiz amacımız bunu tahminlemek)
"""
df_d=df_a.copy()
check_df(df_d)

# Yanlış girilmiş 0 değerlerinin nan şeklinde değiştirilmesi
na_columns=[col for col in df_d.columns if  col not in ["Pregnancies","Outcome","DiabetesPedigreeFunction","Age"]]

for i in na_columns:
    df_d[[i]] = df_d[[i]].replace(0, np.NaN)

# Eksik gözlem birimlerinde Dengesiz veri olduğu için veride Insulin değeri ile aynı anda SkinThickness değeri boş olan Outcome değerleri 0 olanları siliyoruz
drop_list_index=df_d.loc[(df_d["SkinThickness"].isnull()) & (df_d["Insulin"].isnull()) & (df_d["Outcome"]==0) ,["SkinThickness","Outcome","Insulin"]].index
df=df_d.drop(drop_list_index,axis=0)

df.shape

# Kalan Insulin değerlerinin Outcome'a göre gruplayıp medyanlarıyla dolduruyoruz. Diğer değişkenlerinide ortalamaları ile doldurduk
df["Insulin"]=df["Insulin"].fillna(df.groupby("Outcome")["Insulin"].transform("median"))
df=df.apply(lambda x: x.fillna(x.mean()),axis=0)

check_df(df)

# Feature Engineering
df.loc[(df["Age"]>21) &(df["Age"]<35),"NEW_AGE_CAT"]="youngfemale"
df.loc[(df["Age"]>=35) & (df["Age"]<56),"NEW_AGE_CAT"]="maturefemale"
df.loc[(df["Age"]>=56), "NEW_AGE_CAT"]= "seniorfemale"

df.loc[(df["Outcome"]==1) & (df["Age"]<=30),"NEW_DIABETES_TIP"]="Tip_1_Odds"
df.loc[(df["Outcome"]==1) & (df["Age"]>30) & (df["Age"]<50),"NEW_DIABETES_TIP"]="Tip_1_&_2_Odds"
df.loc[(df["Outcome"]==1) & (df["Age"]>=50),"NEW_DIABETES_TIP"]="Tip_2_Odds"

df.loc[(df["SkinThickness"]>30) ,"NEW_IS_DIABETES_EFFECTİVE"]=1
df.loc[(df["SkinThickness"]<=30) ,"NEW_IS_DIABETES_EFFECTİVE"]=0

df.loc[(df['BMI'] <= 18.5), 'NEW_WEIGHT_STATUS'] = 'UnderWeight'
df.loc[(df['BMI'] > 18.5) & (df['BMI'] <= 24.9), 'NEW_WEIGHT_STATUS'] = 'Normal'
df.loc[(df['BMI'] > 24.9) & (df['BMI'] <= 29.9), 'NEW_WEIGHT_STATUS'] = 'Overweight'
df.loc[(df['BMI'] > 29.9 ), 'NEW_WEIGHT_STATUS'] = 'Obese'

# Diabet için risk faktörü olan bazı durumlar var. Bunları sınıflandırarark Değişkenimize değer olarak verdik
df.loc[(df["NEW_WEIGHT_STATUS"]=="Obese") & (df["Outcome"]==0),"NEW_RISK_OF_DIABETES"]="D_Risk_1"
df.loc[(df["NEW_WEIGHT_STATUS"]=="Overweight") & (df["Age"]>45) & (df["Age"]<60),"NEW_RISK_OF_DIABETES"]="D_Risk_2"
df.loc[(df["Age"]>60),"NEW_RISK_OF_DIABETES"]="D_Risk_3"

# Veri setinde Hipertansiyon için bir risk faktörü olan bazı durumlar var.
df.loc[(df["Outcome"]==1),"NEW_RISK_OF_HYPERTENSION"]="H_Risk_1"
df.loc[(df["Age"]>50),"NEW_RISK_OF_HYPERTENSION"]="H_Risk_2"
df.loc[(df["NEW_WEIGHT_STATUS"]=="Obese"),"NEW_RISK_OF_HYPERTENSION"]="H_Risk_3"

df.loc[(df["BloodPressure"]>90),"NEW_IS_HYPERTENSION"]=1
df.loc[(df["BloodPressure"]<=90),"NEW_IS_HYPERTENSION"]=0

df_d.head(50)

# Encoder Islemleri

ohe_cols=[col for col in df.columns if 10>=len(df[col].unique())>2 ]
ohe_cols
df = one_hot_encoder(df, ohe_cols)
df.shape

# CART Model
y=df["Outcome"]
X=df.drop(["Outcome"],axis=1)

#İlk yaptığım model:

cart_model=DecisionTreeClassifier(random_state=17).fit(X,y)
y_pred=cart_model.predict(X)
y_prob=cart_model.predict_proba(X)[:,1]
print(classification_report(y,y_pred))

# HOLDOUT Yöntemi ile model Doğrulama

# Modeli Train Test ayırıyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

# Train hatası
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)
y_pred=cart_model.predict(X_train)
y_prob=cart_model.predict_proba(X_train)[:,1]
print(classification_report(y_train,y_pred))
roc_auc_score(y_train,y_prob)

# Test hatası
y_pred=cart_model.predict(X_test)
y_prob=cart_model.predict_proba(X_test)[:,1] # AUC için prob aldık
print(classification_report(y_test,y_pred))
roc_auc_score(y_test,y_prob)

# Karar ağacını olusturmak
def tree_graph_to_png(tree, feature_names, png_file_to_save):
    tree_str = export_graphviz(tree, feature_names=feature_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(png_file_to_save)

tree_graph_to_png(tree=cart_model, feature_names=X_train.columns, png_file_to_save='cart.png')

# Değişken önem düzeyleri
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(cart_model, X_train)

#Hiperparametre Optimizasyonu ile CART Model oluşturma
"""
DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=None, max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=17, splitter='best')
"""
cart_model=DecisionTreeClassifier(random_state=17)
cart_params={"max_depth":range(5,20),"min_samples_split":[4,6,8,10]}
cart_cv=GridSearchCV(cart_model,cart_params,cv=5,n_jobs=-1,verbose=True)
cart_cv.fit(X_train,y_train)
cart_cv.best_params_

cart_tuned=DecisionTreeClassifier(**cart_cv.best_params_).fit(X_train,y_train)

# train hatası
y_pred=cart_tuned.predict(X_train)
y_prob=cart_tuned.predict_proba(X_train)[:,1]
print(classification_report(y_train,y_pred))
roc_auc_score(y_train,y_prob)

# test hatası
y_pred=cart_tuned.predict(X_test)
y_prob=cart_tuned.predict_proba(X_test)[:,1]
print(classification_report(y_test,y_pred))
roc_auc_score(y_test, y_pred)
roc_auc_score(y_test,y_prob)

