from marshmallow import fields
from app.ext import ma

class SearchSchema(ma.Schema):
    success = fields.Boolean()