import mxnet as mx
import logging

logging.getLogger().setLevel(logging.DEBUG)

# Training data
batch_size = 100
path = '../data/mnist/' # 数据所在的位置
train_iter = mx.io.MNISTIter(image=path+'train-images.idx3-ubyte',
                             label=path+'train-labels.idx1-ubyte',
                             batch_size=batch_size, shuffle=True)
val_iter = mx.io.MNISTIter(image=path+'t10k-images.idx3-ubyte',
                           label=path+'t10k-labels.idx1-ubyte',
                           batch_size=batch_size)
data = mx.sym.var('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5, 5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2, 2), stride=(2, 2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5, 5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2, 2), stride=(2, 2))
# first fullc layer
flatten = mx.sym.flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=128)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

# create a trainable module on CPU
mlp_model = mx.mod.Module(symbol=lenet, context=mx.cpu())
mlp_model.fit(train_iter,
              eval_data=val_iter,
              optimizer='sgd',
              optimizer_params={'learning_rate': 0.1},
              eval_metric='acc',
              batch_end_callback=mx.callback.Speedometer(batch_size, 500),
              num_epoch=10)
acc = mx.metric.Accuracy()
mlp_model.score(val_iter, acc)
print(acc)