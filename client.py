import requests

HOST = 'http://127.0.0.1:5000/upscale'

img_1 = 'lama_300px.png'
img_2 = 'lama_600px.png'

response1 = requests.post(HOST, files={'image_1': img_1, 'image_2': img_2})
task_id = response1.json()['task_id']

status = 'PENDING'
result = None
while status == 'PENDING':
    response = requests.get(f'{HOST}/{task_id}').json()
    status = response['status']
    result = response['result']

print(result)