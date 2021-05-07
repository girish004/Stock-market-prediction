from django import forms  
class StudentForm(forms.Form):  
    open = forms.CharField(label="Enter Open value",max_length=50)
    high  = forms.CharField(label="Enter high value", max_length = 100)
    low  = forms.CharField(label="Enter low value", max_length = 100)
    adjacent = forms.CharField(label="Enter Adjacent close value", max_length = 100)
    vol  = forms.CharField(label="Enter volume", max_length = 100)