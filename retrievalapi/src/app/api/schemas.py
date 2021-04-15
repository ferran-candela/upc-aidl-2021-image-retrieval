from marshmallow import fields
from ..ext import ma

class SearchSchema(ma.Schema):
    ranking = fields.List(fields.Integer())
    success = fields.Boolean()

class ModelSchema(ma.Schema):
    models = fields.List(fields.String())