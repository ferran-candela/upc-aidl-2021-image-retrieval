from marshmallow import fields
from ..ext import ma

class SearchSchema(ma.Schema):
    success = fields.Boolean()