from django.db import models



class regtable(models.Model):
    name=models.CharField(max_length=150)
    phone_number=models.CharField(max_length=120)
    email=models.CharField(max_length=120)
    password=models.CharField(max_length=120) 

class fileuploadtable(models.Model):
    image=models.FileField(max_length=150)
    name=models.CharField(max_length=150)
    u_id=models.CharField(max_length=120)
    result=models.CharField(max_length=120)
