from django.shortcuts import render
import pickle
import xgboost
import pandas as pd
import numpy as np

np.random.seed(70)

f = open("model_pkg.pkl", 'rb')
pkg_model = pickle.load(f)

data = pd.read_csv('dataset.csv')

data.drop(columns=['OE1', 'Disc_Elective_1_Grade'], inplace=True)
x = data.iloc[:, :-1]
y = data.iloc[:, -1]

from sklearn.model_selection import train_test_split
import xgboost

model_xgboost = xgboost.XGBClassifier(learning_rate=0.1,
                                      max_depth=5,
                                      n_estimators=5000,
                                      subsample=0.5,
                                      colsample_bytree=0.5,
                                      eval_metrics='auc',
                                      verbosity=1)

X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2)
eval_set = [(X_valid, y_valid)]

model_xgboost.fit(X_train,
                  y_train,
                  early_stopping_rounds=10,
                  eval_set=eval_set,
                  verbose=True)

y_train_pred = model_xgboost.predict(X_train)
y_valid_pred = model_xgboost.predict(X_valid)


def index(request):
    return render(request, 'index.html')


def single(request):
    return render(request, 'single.html')


def predict_single(request):
    # f = open("final_xgboost.pkl", 'rb')
    # model = pickle.load(f)
    cgpa_6 = float(request.GET.get('cgpa_6'))
    toc_grade = float(request.GET.get('toc_grade'))
    sgpa_5 = float(request.GET.get('sgpa_5'))
    os_grade = float(request.GET.get('os_grade'))
    cd_grade = float(request.GET.get('cd_grade'))
    cg_grade = float(request.GET.get('cg_grade'))
    oe_grade = float(request.GET.get('oe_grade'))
    coa_grade = float(request.GET.get('coa_grade'))
    # oe_name = request.GET.get('oe_name')

    pred = model_xgboost.predict_proba(
        np.array([[cgpa_6, toc_grade, sgpa_5, os_grade, cd_grade, cg_grade, oe_grade, coa_grade]]))[0][0]

    is_placed = round(pred)
    # print('*' * 20, is_placed)
    package = pkg_model.predict(np.array([[is_placed, cgpa_6, toc_grade, sgpa_5, os_grade]]))[0]

    return render(request, 'singleOutput.html', {'pred': round(pred * 100,2), 'package': abs(round(package, 2))})

#
# def index(request):
#     return render(request, 'index.html')
