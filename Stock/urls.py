from django.urls import path
from .views import *

urlpatterns = [
    path('', homePageView, name='home'),
    path('hello/',hello,name='hello'),
    path('form/',formss,name='forms'),
]