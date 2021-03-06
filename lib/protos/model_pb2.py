# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: lib/protos/model.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from lib.protos import faster_rcnn_pb2 as lib_dot_protos_dot_faster__rcnn__pb2
from lib.protos import ssd_pb2 as lib_dot_protos_dot_ssd__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='lib/protos/model.proto',
  package='lib.protos',
  syntax='proto2',
  serialized_pb=_b('\n\x16lib/protos/model.proto\x12\nlib.protos\x1a\x1clib/protos/faster_rcnn.proto\x1a\x14lib/protos/ssd.proto\"h\n\x0e\x44\x65tectionModel\x12-\n\x0b\x66\x61ster_rcnn\x18\x01 \x01(\x0b\x32\x16.lib.protos.FasterRcnnH\x00\x12\x1e\n\x03ssd\x18\x02 \x01(\x0b\x32\x0f.lib.protos.SsdH\x00\x42\x07\n\x05model')
  ,
  dependencies=[lib_dot_protos_dot_faster__rcnn__pb2.DESCRIPTOR,lib_dot_protos_dot_ssd__pb2.DESCRIPTOR,])




_DETECTIONMODEL = _descriptor.Descriptor(
  name='DetectionModel',
  full_name='lib.protos.DetectionModel',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='faster_rcnn', full_name='lib.protos.DetectionModel.faster_rcnn', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='ssd', full_name='lib.protos.DetectionModel.ssd', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='model', full_name='lib.protos.DetectionModel.model',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=90,
  serialized_end=194,
)

_DETECTIONMODEL.fields_by_name['faster_rcnn'].message_type = lib_dot_protos_dot_faster__rcnn__pb2._FASTERRCNN
_DETECTIONMODEL.fields_by_name['ssd'].message_type = lib_dot_protos_dot_ssd__pb2._SSD
_DETECTIONMODEL.oneofs_by_name['model'].fields.append(
  _DETECTIONMODEL.fields_by_name['faster_rcnn'])
_DETECTIONMODEL.fields_by_name['faster_rcnn'].containing_oneof = _DETECTIONMODEL.oneofs_by_name['model']
_DETECTIONMODEL.oneofs_by_name['model'].fields.append(
  _DETECTIONMODEL.fields_by_name['ssd'])
_DETECTIONMODEL.fields_by_name['ssd'].containing_oneof = _DETECTIONMODEL.oneofs_by_name['model']
DESCRIPTOR.message_types_by_name['DetectionModel'] = _DETECTIONMODEL
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

DetectionModel = _reflection.GeneratedProtocolMessageType('DetectionModel', (_message.Message,), dict(
  DESCRIPTOR = _DETECTIONMODEL,
  __module__ = 'lib.protos.model_pb2'
  # @@protoc_insertion_point(class_scope:lib.protos.DetectionModel)
  ))
_sym_db.RegisterMessage(DetectionModel)


# @@protoc_insertion_point(module_scope)
