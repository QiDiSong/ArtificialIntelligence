# TensorFlow Lite 指南





TensorFlow Lite 是一组工具，可帮助开发者在移动设备、嵌入式设备和 IoT 设备上运行 TensorFlow 模型。它支持设备端机器学习推断，延迟较低，并且二进制文件很小。

TensorFlow Lite 包括两个主要组件：

- [TensorFlow Lite 解释器](https://www.tensorflow.org/lite/guide/inference?hl=zh-cn)，它可在手机、嵌入式 Linux 设备和微控制器等很多不同类型的硬件上运行经过专门优化的模型。
- [TensorFlow Lite 转换器](https://www.tensorflow.org/lite/convert/index?hl=zh-cn)，它可将 TensorFlow 模型转换为方便解释器使用的格式，并可引入优化以减小二进制文件的大小和提高性能。

### 边缘机器学习

TensorFlow Lite 旨在让您轻松地在网络“边缘”的设备上执行机器学习，而无需在设备与服务器之间来回发送数据。对开发者来说，在设备端执行机器学习有助于：

- 缩短延迟：数据无需往返服务器
- 保护隐私：任何数据都不会离开设备
- 减少连接：不需要互联网连接
- 降低功耗：网络连接非常耗电

TensorFlow Lite 支持各种设备，从超小的微控制器到功能强大的手机，不一而足。
