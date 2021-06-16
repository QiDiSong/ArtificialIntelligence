## Dropout

> Dropout是在每次神经网络的训练过程中，使得部分神经元工作而另外一部分神经元不工作。而测试的时候激活所有神经元，用所有的神经元进行测试。这样便可以有效的缓解过拟合，提高模型的准确率。

```

def neural_network(x):

    hidden_layer_1 = tf.add(tf.matmul(x, weights['hidden_1']), biases['b1'])

    L1 = tf.nn.tanh(hidden_layer_1)

    dropout1 = tf.nn.dropout(L1,0.5)

    out_layer = tf.matmul(dropout1, weights['out']) + biases['out']

    return out_layer
```

