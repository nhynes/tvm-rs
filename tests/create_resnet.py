import nnvm
import nnvm.compiler
import nnvm.testing

net, _ = nnvm.testing.resnet.get_workload(batch_size=1, image_shape=(3, 224, 224))
graph = nnvm.graph.create(net)
nnvm.compiler.graph_util.infer_shape(graph, data=(1, 3, 224, 224))

with open('resnet_inference.json', 'w') as f_resnet:
    f_resnet.write(graph.apply('InferShape').json())
