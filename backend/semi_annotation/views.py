# from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.http import FileResponse
import zipfile
import json
import os


def get_csrf_cookie(request):
    csrf_token = get_token(request)
    return JsonResponse({'csrfToken': csrf_token})


@csrf_exempt
def upload_files(request):
    if request.method == 'POST':
        files = request.FILES.getlist('files')
        for file in files:
            with open('media/' + file.name, 'wb+') as destination:
                for chunk in file.chunks():
                    destination.write(chunk)
        return HttpResponse('Files uploaded successfully')
    return HttpResponse('Invalid request method')


def cluster_data(request):
    return JsonResponse({"message": "Completed"})


def download_data(request):
    zip_filename = 'files.zip'
    zip_file_path = os.path.join('tmp/', zip_filename)

    with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
        for root, directories, files in os.walk('annotated_data/'):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, 'annotated_data/')  # Calculate the relative path for preserving directories in the zip
                zip_file.write(file_path, arcname=arcname)

    response = FileResponse(open(zip_file_path, 'rb'), content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename="{0}"'.format(zip_filename)

    os.remove(zip_file_path)

    return response


def get_image(request):
    image_path = os.path.join('media/', 'image.png')
    with open(image_path, 'rb') as image_file:
        response = HttpResponse(image_file.read(), content_type='image/jpeg')
        response['Content-Disposition'] = 'attachment; filename=image.jpg'
        
    return response


@csrf_exempt
def submit_answer(request):
    if request.method == 'POST':
        try:
            body = json.loads(request.body)
            answer = body.get('answer', '')
            # Process the answer and provide a response
            # TODO
            response_message = 'Your answer was: ' + answer  # Replace with your answer processing logic

            return JsonResponse({'message': response_message})
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON format'})

    return JsonResponse({'error': 'Invalid request method'})
