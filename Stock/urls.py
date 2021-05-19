from django.urls import path
from .views import *
from django.views.generic import TemplateView
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
urlpatterns = [
    path('', homePageView, name='home'),
    path('hello/',hello,name='hello'),
    path('form/',formss,name='forms'),
    path('logo.png/',logo,name='logo'),
    path('cht.js/',js,name='js'),
]
urlpatterns+=staticfiles_urlpatterns()
