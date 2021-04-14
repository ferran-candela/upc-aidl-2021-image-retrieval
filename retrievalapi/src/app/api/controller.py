import os
from flask import request, Blueprint, jsonify, safe_join, send_file
from flask_restful import Api, Resource, reqparse
from werkzeug.utils import secure_filename
from .schemas import SearchSchema, ModelSchema
from imageretrieval.src.engine import RetrievalEngine
from imageretrieval.src.config import DeviceConfig, FoldersConfig

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

api_bp = Blueprint('api_bp', __name__)

api = Api(api_bp)

search_schema = SearchSchema()
model_schema = ModelSchema()

device = DeviceConfig.DEVICE

retrieval_engine = RetrievalEngine(device, FoldersConfig.WORK_DIR)
retrieval_engine.load_models_and_precomputed_features()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class SearchResource(Resource):
    def post(self):
        try:
            model_name = request.form.get('model')
            topK = int(request.form.get('topK'))
            file = request.files['image']
            if file and allowed_file(file.filename):
                # From flask uploading tutorial
                filename = secure_filename(file.filename)
                tmpFile = os.path.join(UPLOAD_FOLDER, filename)
                file.save(tmpFile)
                ranking = retrieval_engine.query(model_name, tmpFile, topK)
                resp = search_schema.dump({
                    'ranking': ranking,
                    'success': True 
                    })
                os.remove(tmpFile)
                return resp
            else:
                # return error
                resp = search_schema.dump({ 'success': False })
                return resp
        except Exception as e:
            print('\n', e)
            # return error
            resp = search_schema.dump({ 'success': False })
            return resp

class ModelResource(Resource):
    def get(self):
        models = retrieval_engine.get_model_names()
        return model_schema.dump({
                        'models': models
                        })

@api.representation('image/jpeg')
def output_file(data, code, headers):
    filepath = data["filePath"]

    response = send_file(
        filename_or_fp=filepath,
        mimetype="image/jpeg",
        as_attachment=True,
        attachment_filename=data["fileName"]
    )
    return response

class ImageResource(Resource):
    def get(self, id):
        image_path = retrieval_engine.get_image_path(id)

        return {
            "filePath": image_path,
            "fileName": id + '.jpg'
        }

api.add_resource(SearchResource, '/api/search', endpoint='search_resource')
api.add_resource(ModelResource, '/api/models', endpoint='model_resource')
api.add_resource(ImageResource, '/api/images/<string:id>', endpoint='images_resource')
