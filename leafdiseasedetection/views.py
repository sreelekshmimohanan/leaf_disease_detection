from django.http import HttpResponse
from django.shortcuts import render
from django.shortcuts import redirect
# FILE UPLOAD AND VIEW
from  django.core.files.storage import FileSystemStorage
# SESSION
from django.conf import settings
from .models import *
from ML import test1
import os
from tensorflow.keras import backend as K

def first(request):
    return render(request,'index.html')
def index(request):
    return render(request,'index.html')
def addregister(request):
    if request.method=="POST":
        a=request.POST.get('name')
        b=request.POST.get('phone_number')
        c=request.POST.get('email')
        d=request.POST.get('password')
        e=regtable(name=a,phone_number=b,email=c,password=d)
        e.save()
    return redirect(login) 

def register(request):
    return render(request,'register.html')

def login(request):
    return render(request,'login.html')

def addlogin(request):
    email = request.POST.get('email')
    password = request.POST.get('password')
    if email == 'admin@gmail.com' and password =='admin':
        request.session['admin'] = 'admin'
        return render(request,'index.html')

    elif regtable.objects.filter(email=email,password=password).exists():
            userdetails=regtable.objects.get(email=request.POST['email'], password=password)
            request.session['uid'] = userdetails.id
            return render(request,'index.html')

    else:
        return render(request, 'login.html', {'message':'Invalid Email or Password'})
    

def fileupload(request):      
    return render(request,'fileupload.html')

def v_users(request):
    user=regtable.objects.all()
    return render(request,'viewusers.html',{'result':user})

def test(request):
    return render(request,'test.html')

def addfile(request):
    if request.method == "POST":
        u_id = request.session['uid']

        file = request.FILES['image']
        try:
            os.remove("media/input/test/test.jpg")
        except:
            pass

        fs = FileSystemStorage(location="media/input/test")
        fs.save("test.jpg", file)
        fs = FileSystemStorage()

        # Clear session before making predictions
        K.clear_session()

        result = test1.predict()

        # Do not clear session again after prediction
        # K.clear_session()

        cus = fileuploadtable(u_id=u_id, result=result, image=file)
        cus.save()

    return redirect(viewresult)

def viewresult(request):
    uid=request.session['uid']
    user = fileuploadtable.objects.filter(u_id=uid)
    return render(request, 'viewresult.html', {'result': user})

def viewresult_admin(request):
    user = fileuploadtable.objects.all()
    user1 = regtable.objects.all()
    for i in user:
        for j in user1:
            if str(i.u_id) == str(j.id):
                i.u_id = j.name

    return render(request,'viewresult_admin.html', {'result': user})

def logout(request):
    session_keys=list(request.session.keys())
    for key in session_keys:
        del request.session[key]
    return redirect(first)