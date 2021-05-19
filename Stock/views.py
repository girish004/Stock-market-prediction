from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.forms import *
from Stock.forms import *
# Create your views here.
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from openpyxl import Workbook
import xlsxwriter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import requests
from bs4 import BeautifulSoup
import bs4
from fastnumbers import isfloat
from fastnumbers import fast_float
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
import seaborn as sns
from requests.exceptions import HTTPError
import json
from bs4 import BeautifulSoup
from html.parser import HTMLParser
from selenium import webdriver
from sklearn.preprocessing import MinMaxScaler
def get_table_simple(table,is_table_tag=True):
    elems = table.find_all('tr') if is_table_tag else get_children(table)
    table_data = list()
    for row in elems:
        row_data = list()
        row_elems = get_children(row)
        for elem in row_elems:
            text = elem.text.strip().replace("\n","")
            text = remove_multiple_spaces(text)
            if len(text)==0:
                continue
            row_data.append(text)
        table_data.append(row_data)
    return table_data

def remove_multiple_spaces(string):
    if type(string)==str:
        return ' '.join(string.split())
    return string
def get_children(html_content):
    return [item for item in html_content.children if type(item)==bs4.element.Tag or len(str(item).replace("\n","").strip())>0]
def data():
    driver = webdriver.Chrome("C:/Users/HP/chromedriver.exe")
    url = "https://finance.yahoo.com/quote/%5ENSEI/history?p=%5ENSEI"
    driver.get(url)
    html = driver.page_source
    page_content = BeautifulSoup(html, "html.parser")
    table_rows=page_content.find('tr',attrs={'class':'BdT Bdc($seperatorColor) Ta(end) Fz(s) Whs(nw)'}) #.text.split(",")
    lis=[]
    table_string="<table><tr>"
    for row in table_rows:
        table_string+=str(row)
    table_string=table_string+"</tr></table>"
    head=['Date', 'Open', 'High', 'Low', 'Close', 'Adj close', 'Volume']
    content = BeautifulSoup(table_string,"html.parser")
    temp_dataset=get_table_simple(content)
    dataset=[]
    temp_dataset=temp_dataset[0]
    print(temp_dataset)
    dataset=temp_dataset
    print(dataset)
    for i in range(len(dataset)):
        if i!=0 and dataset[i]!='-':
            dataset[i] =float(dataset[i].replace(',', ''))
    pd_xl_file = pd.ExcelFile("D:\Girish\AI\Project\stock_data.xlsx")
    df = pd_xl_file.parse("Sheet1")
    bo=False
    if df._get_value(df.shape[0]-1, 0, takeable = True)!=dataset[0]:
        df.loc[df.shape[0]+1]=dataset
        bo=True
    writer = pd.ExcelWriter('D:\Girish\AI\Project\stock_data.xlsx', engine='xlsxwriter')
# Write data to an excel
    df.to_excel(writer,sheet_name="Sheet1",index=False)
    writer.close()
    return bo
def dl():
    dataset = pd.read_excel ('D:\Girish\AI\Project\stock_data.xlsx') #place "r" before the path string to address special character, such as '\'. Don't forget to put the file name at the end of the path + '.xlsx'
    dataset=dataset[["Date","Open","High","Low","Close","Adj close","Volume"]]
    datelist=dataset["Date"].tolist()
    openlist=dataset["Open"].tolist()
    highlist=dataset["High"].tolist()
    lowlist=dataset["Low"].tolist()
    adlist=dataset["Adj close"].tolist()
    vollist=dataset["Volume"].tolist()
    leng=len(datelist)
    open2list=[["Date","Open"]]
    high2list=[["Date","High"]]
    low2list=[["Date","Low"]]
    ad2list=[["Date","Adjacent"]]
    vol2list=[["Date","Volume"]]
    for i in range(0,len(datelist)):
        open2list.append([datelist[i],openlist[i]])
        high2list.append([datelist[i],highlist[i]])
        low2list.append([datelist[i],lowlist[i]])
        ad2list.append([datelist[i],adlist[i]])
        vol2list.append([datelist[i],vollist[i]])
    scalerx = MinMaxScaler(feature_range=(0,1))
    scalery = MinMaxScaler(feature_range=(0,1))
    scaled_data = dataset.iloc[:, [1, 2, 3, 4, 5, 6]]
    scaled_data=pd.DataFrame(scaled_data,columns=["Open","High","Low","Close","Adj close","Volume"])
    testing_set=scaled_data.iloc[1500:, [0, 1, 2, 3, 4, 5]]
    X=scaled_data.iloc[:, [0, 1, 2, 4, 5]]
    X=scalerx.fit_transform(X)
    y=scaled_data.iloc[:,[3]]
    y=scalery.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=0,shuffle=False)
    X_train, X_test, y_train, y_test=np.array(X_train),np.array(X_test),np.array(y_train),np.array(y_test)
    testing_set=np.array(testing_set)
    testing_set=np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1 ))
    model=Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    sc = StandardScaler()
    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    # Train the model
    model.fit(X_train, y_train, batch_size=1, epochs=1)
    predictions=model.predict(X_test)
    predictions=pd.DataFrame(predictions)
    y_test=pd.DataFrame(y_test)
    y_test=scalery.inverse_transform(y_test)
    predictions=scalery.inverse_transform(predictions)
    for i in range(len(y_test)):
        print(y_test[i],predictions[i])
    return scalerx,scalery,scaled_data,X_test,model,datelist,open2list,high2list,low2list,ad2list,vol2list

scalerx,scalery,scaled_data,X_test,model,datelist,open2list,high2list,low2list,ad2list,vol2list=dl()
def homePageView(request):
    return HttpResponse('Hello, World!')

def formss(request):
    global X_test,ques,y_test,scalerx,scalery,scaled_data,X_test,model,datelist,open2list,high2list,low2list,ad2list,vol2list
    if data():
       scalerx,scalery,scaled_data,X_test,model,datelist,open2list,high2list,low2list,ad2list,vol2list=dl()
    if(request.method=='POST'):
        student=StudentForm(request.POST)
        if(student.is_valid()):
            open=student.cleaned_data.get("open")
            high=student.cleaned_data.get("high")
            low=student.cleaned_data.get("low")
            adjacent=student.cleaned_data.get("adjacent")
            vol=student.cleaned_data.get("vol")
            X_testt=scaled_data.iloc[:, [0, 1, 2, 4, 5]]
            y_testt=scaled_data.iloc[:,[3]]
            X_testt.loc[len(X_test)]=[open,high,low,adjacent,vol]
            scalerxl = MinMaxScaler(feature_range=(0,1))
            X_testt=scalerxl.fit_transform(X_testt)
            X_trainn, X_testt, y_trainn, y_testt = train_test_split(X_testt, y_testt, test_size=0.10, random_state=0,shuffle=False)
            X_testt=np.array(X_testt)
            X_testt=np.reshape(X_testt, (X_testt.shape[0], X_testt.shape[1], 1))
            ans=model.predict(X_testt)
            ans=scalery.inverse_transform(ans)
            ans[0][0]=ans[len(ans)-1][0]
            print(open,high,low,adjacent,vol,ans)
    else:
        student = StudentForm()
        ans=[[0]]
    passd={'form':student,
    'ans':ans,
    'date':datelist,'open':open2list,'high':high2list,'low':low2list,'adj':ad2list,'vol':vol2list}
    return render(request,"Stock/index.html/",passd)
def hello(request):
    return render(request,"Stock/index.html/")
def logo(request):
    return render(request,"Stock/logo.png/")
def js(request):
    return render(request,"Stock/cht.js/")
