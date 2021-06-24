from django.urls import path, re_path
from . import views


urlpatterns = [
        path('', views.mltester, name='mltester'),
        path('linear_regression/', views.linear_regression, name='linear_regression'),
        path('logistic_regression/', views.logistic_regression, name='logistic_regression'),
        path('svm/', views.svm, name='svm'),
        path('decision_tree/', views.decision_tree, name='decision_tree'),
        path('custom_neural_network/', views.custom_neural_network, name='custom_neural_network')

]