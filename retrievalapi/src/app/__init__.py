from flask import Flask, jsonify
from flask_restful import Api
from .common.error_handling import AppErrorBaseClass
from .api.controller import api_bp
from .ext import ma

def create_app(settings_module):
    app = Flask(__name__)
    app.config.from_object(settings_module)
    # Init extensions
    ma.init_app(app)
    # Catch 404
    Api(app, catch_all_404s=True)
    # Disable stric mode using ending /
    app.url_map.strict_slashes = False
    # Register blueprints
    app.register_blueprint(api_bp)
    # Register custom error habdler
    register_error_handlers(app)
    return app

def register_error_handlers(app):
    @app.errorhandler(Exception)
    def handle_exception_error(e):
        return jsonify({'msg': 'Internal server error'}), 500
    @app.errorhandler(405)
    def handle_405_error(e):
        return jsonify({'msg': 'Method not allowed'}), 405
    @app.errorhandler(403)
    def handle_403_error(e):
        return jsonify({'msg': 'Forbidden error'}), 403
    @app.errorhandler(404)
    def handle_404_error(e):
        return jsonify({'msg': 'Not Found error'}), 404
    @app.errorhandler(AppErrorBaseClass)
    def handle_app_base_error(e):
        return jsonify({'msg': str(e)}), 500