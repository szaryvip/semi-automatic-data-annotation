from django.urls import path
from django.views.decorators.csrf import csrf_exempt

from . import views

urlpatterns = [
    path('csrf_cookie/', csrf_exempt(views.get_csrf_cookie), name='csrf_cookie'),
    path('upload/', views.upload_files, name='upload'),
    path('cluster_data/', views.cluster_data, name='cluster'),
    path('download_files/', views.download_data, name='download'),
    path('get_image/', views.get_image, name='image'),
    path('submit_answer/', views.submit_answer, name='answer'),
    
]