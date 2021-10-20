from django.http import HttpResponse
from django.shortcuts import render
import joblib
import numpy
import numpy as np

def home(request):
    return render(request,"home.html")

def result(request):

    clf = joblib.load('final_model.sav')

    lis = []

    lis.append(int(request.GET['fever']))
    lis.append(int(request.GET['bodypain']))
    lis.append(int(request.GET['age']))
    lis.append(int(request.GET['runnynose']))
    lis.append(int(request.GET['diffbreathing']))

    print(lis)

    ans = clf.predict_proba([lis])[0][1]
    ans = round(ans*100)

    return render(request,"result.html",{'ans':ans})

