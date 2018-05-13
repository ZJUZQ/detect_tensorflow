
### Protobuf Compilation

This Tensorflow Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled. 

```
/home/zq/3rdparty/protoc_3.5/bin/protoc lib/protos/*.proto --python_out=.
```