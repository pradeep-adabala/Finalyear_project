from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('user/', views.User, name='user_login'),
    path('admin_login/', views.AdminLogin, name='admin_login'),  # Renamed to avoid conflict
    path('admin_screen/', views.GenerateModel, name='admin_screen'),
    path('generate_model/', views.GenerateModel, name='generate_model'),
    path('view_train/', views.ViewTrain, name='view_train'),
    path('user_check/', views.UserCheck, name='user_check'),
]
