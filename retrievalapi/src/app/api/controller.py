import os
from flask import request, Blueprint, jsonify
from flask_restful import Api, Resource, reqparse
from werkzeug.utils import secure_filename
from .schemas import SearchSchema

UPLOAD_FOLDER = '/tmp'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

api_bp = Blueprint('api_bp', __name__)

api = Api(api_bp)

search_schema = SearchSchema()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class SearchResource(Resource):
    def post(self):
        file = request.files['file']
        if file and allowed_file(file.filename):
            # From flask uploading tutorial
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            resp = search_schema.dump({ 'success': True })
            return resp
        else:
            # return error
            resp = search_schema.dump({ 'success': False })
            return resp

api.add_resource(SearchResource, '/api/search', endpoint='search_resource')
