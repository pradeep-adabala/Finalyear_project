# views.py

from django.shortcuts import render 
from django.http import HttpResponse
import pandas as pd
import tensorflow as tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Activation, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore
from sklearn.preprocessing import OneHotEncoder

global model

def index(request):
    return render(request, 'profile_detection/index.html')

def User(request):
    if request.method == 'GET':
        return render(request, 'profile_detection/user.html')

def AdminLogin(request):
    return render(request, 'profile_detection/admin_login.html')

def Admin(request):
    if request.method == 'POST':
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        if username == 'admin' and password == 'admin':
            context = {'data': 'welcome ' + username}
            return render(request, 'profile_detection/adminscreen.html', context)
        else:
            context = {'data': 'login failed'}
            return render(request, 'profile_detection/admin_login.html', context)

def importdata():
    balance_data = pd.read_csv('C:/FakeProfile/Profile/dataset/dataset.txt')
    balance_data = balance_data.abs()
    return balance_data

def splitdataset(balance_data):
    X = balance_data.values[:, 0:8]
    y_ = balance_data.values[:, 8]
    y_ = y_.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(y_)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    return train_x, test_x, train_y, test_y

def UserCheck(request):
    if request.method == 'POST':
        data = request.POST.get('t1', False)
        input = 'Account_Age,Gender,User_Age,Link_Desc,Status_Count,Friend_Count,Location,Location_IP\n'
        input += data + "\n"
        with open("C:/FakeProfile/Profile/dataset/test.txt", "w") as f:
            f.write(input)
        test = pd.read_csv('C:/FakeProfile/Profile/dataset/test.txt')
        test = test.values[:, 0:8]
        predict = model.predict_classes(test)
        msg = 'Given Account Details Predicted As Genuine' if str(predict[0]) == '0' else 'Given Account Details Predicted As Fake'
        context = {'data': msg}
        return render(request, 'profile_detection/user.html', context)

def GenerateModel(request):
    global model
    data = importdata()
    train_x, test_x, train_y, test_y = splitdataset(data)
    model = Sequential()
    model.add(Dense(200, input_shape=(8,), activation='relu', name='fc1'))
    model.add(Dense(200, activation='relu', name='fc2'))
    model.add(Dense(2, activation='softmax', name='output'))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_x, train_y, verbose=2, batch_size=5, epochs=200)
    results = model.evaluate(test_x, test_y)
    ann_acc = results[1] * 100
    context = {'data': 'ANN Accuracy : ' + str(ann_acc)}
    return render(request, 'profile_detection/adminscreen.html', context)

def ViewTrain(request):
    data = pd.read_csv('C:/FakeProfile/Profile/dataset/dataset.txt')
    strdata = '<table border=1 align=center width=100%><tr><th>Account Age</th><th>Gender</th><th>User Age</th><th>Link Description</th><th>Status Count</th><th>Friend Count</th><th>Location</th><th>Location IP</th><th>Profile Status</th></tr><tr>'
    for i in range(len(data)):
        strdata += ''.join([f'<td>{data.iloc[i,j]}</td>' for j in range(len(data.columns))]) + '</tr><tr>'
    context = {'data': strdata}
    return render(request, 'profile_detection/viewdata.html', context)