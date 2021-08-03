**End to end**：指的是输入原始数据，输出的是最后结果，应用在特征学习融入算法，无需单独处理。

**end-to-end（端对端）的方法，一端输入我的原始数据，一端输出我想得到的结果。只关心输入和输出，中间的步骤全部都不管。**

　　端到端指的是输入是原始数据，输出是最后的结果，原来输入端不是直接的原始数据，而是在原始数据中提取的特征，这一点在图像问题上尤为突出，因为图像像素数太多，数据维度高，会产生维度灾难，所以原来一个思路是手工提取图像的一些关键特征，这实际就是就一个降维的过程。
　　那么问题来了，特征怎么提？
　　特征提取的好坏异常关键，甚至比学习算法还重要，举个例子，对一系列人的数据分类，分类结果是性别，如果你提取的特征是头发的颜色，无论分类算法如何，分类效果都不会好，如果你提取的特征是头发的长短，这个特征就会好很多，但是还是会有错误，如果你提取了一个超强特征，比如染色体的数据，那你的分类基本就不会错了。
　　这就意味着，特征需要足够的经验去设计，这在数据量越来越大的情况下也越来越困难。
　　于是就出现了端到端网络，特征可以自己去学习，所以**特征提取这一步也就融入到算法当中，不需要人来干预了**。

　　简单来说就是**深度神经网络处理问题不需要像传统模型那样**，**如同生产线般一步步去处理输入数据直至输出最后的结果**（其中每一步处理过程都是经过人为考量设定好的 (“hand-crafted” function)）。

　　与之相反，**只需给出输入数据以及输出，神经网络就可以通过训练自动“学得”之前那些一步接一步的 “hand-crafted” functions**。

### 相关理解：

**1、传统系统需要几个模块串行分别设计，end2end把中间模块都去掉了。**
以机器翻译为例 要设计翻译模型 语言模型 调序模型
端到端就是**直接一个模型**搞定

2、**cnn就是比较典型的end2end模型**。在图像分类里输入image各通道像素，输出图像类别。 相比于非end2end，conv层的卷积核可以充当feature extractor部分而不需要额外的工作去做特征工程的内容。尽管每一层需要自己设计，但如何得到feature并不需要额外的操作。

3、另一种理解：就是输入一头猪，输出的是香肠

### End-to-end在不同应用场景下有不同的具体诠释，

对于视觉领域而言，end-end一词多用于基于视觉的机器控制方面，具体表现是，神经网络的输入为原始图片，神经网络的输出为（可以直接控制机器的）控制指令，如：

\1. Nvidia的基于CNNs的end-end自动驾驶，输入图片，直接输出steering angle。从视频来看效果拔群，但其实这个系统目前只能做简单的follow lane，与真正的自动驾驶差距较大。亮点是证实了end-end在自动驾驶领域的可行性，并且对于数据集进行了augmentation。链接：[https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/](https://link.zhihu.com/?target=https%3A//devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

\2. Google的paper: Learning Hand-Eye Coordination for Robotic Grasping with Deep Learning and Large-Scale Data Collection，也可以算是end-end学习：输入图片，输出控制机械手移动的指令来抓取物品。这篇论文很赞，推荐：[https://arxiv.org/pdf/1603.02199v4.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1603.02199v4.pdf)

\3. DeepMind神作Human-level control through deep reinforcement learning，其实也可以归为end-end，深度增强学习开山之作，值得学习：[http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html](https://link.zhihu.com/?target=http%3A//www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

\4. Princeton大学有个Deep Driving项目，介于end-end和传统的model based的自动驾驶之间，输入为图片，输出一些有用的affordance（实在不知道这词怎么翻译合适…）例如车身姿态、与前车距离、距路边距离等，然后利用这些数据通过公式计算所需的具体驾驶指令如加速、刹车、转向等。链接：[http://deepdriving.cs.princeton.edu/](https://link.zhihu.com/?target=http%3A//deepdriving.cs.princeton.edu/)

### 总之

end-end不是什么新东西，也不是什么神奇的东西，**仅仅是直接输入原始数据，直接输出最终目标的一种思想。**
