import os
import uuid

import flask
from celery import Celery
from celery.result import AsyncResult
from flask import Flask, jsonify, request, abort

import cv2
from cv2 import dnn_superres
from flask.views import MethodView


app_name = 'app'
app = Flask(app_name)
app.config['UPLOAD_FOLDER'] = 'files'
celery = Celery(app_name, broker='redis://localhost:6379/1', backend='redis://localhost:6379/2')
celery.conf.update(app.config)


class ContextTask(celery.Task):
    def __call__(self, *args, **kwargs):
        with app.app_context():
            return self.run(*args, *kwargs)


celery.Task = ContextTask


@celery.task
def upscale(input_path: str, output_path: str, model_path: str = 'EDSR_x2.pb') -> None:
    """
    :param input_path: путь к изображению для апскейла
    :param output_path:  путь к выходному файлу
    :param model_path: путь к ИИ модели
    :return:
    """
    scaler = dnn_superres.DnnSuperResImpl_create()
    scaler.readModel(model_path)
    scaler.setModel("edsr", 2)
    image = cv2.imread(input_path)
    result = scaler.upsample(image)
    cv2.imwrite(output_path, result)


class Upscale(MethodView):

    def get(self, task_id):
        task = AsyncResult(task_id, app=celery)
        return jsonify({'status': task.status,
                        'result': task.result})

    def post(self):
        image_pathes = [self.save_image(field) for field in ('image_1', 'image_2')]
        task = upscale.delay(*image_pathes)
        return jsonify({'task_id': task.id})

    def save_image(self, field):
        image = request.files.get(field)
        extension = image.filename.split('.')[-1]
        path = os.path.join('files', f'{uuid.uuid4()}.{extension}')
        image.save(path)
        return path


@app.route('/processed/<path:file>', methods=['GET'])
def get_file(file):
    try:
        return flask.send_file(file, as_attachment=True)
    except FileNotFoundError:
        abort(404)


upscale_view = Upscale.as_view('upscale')
app.add_url_rule('/upscale', view_func=upscale_view, methods=['POST'])
app.add_url_rule('/tasks/<string:task_id>', view_func=upscale_view, methods=['GET'])

# task = upscale.delay('lama_300px.png', 'lama_600px.png')

if __name__ == '__main__':
    app.run()
