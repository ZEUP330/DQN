# Deep Q Network

## DRL
原因：在普通的Q-learning中，当状态和动作空间是离散且维数不高时可使用Q-Table储存每个状态动作对的Q值，而当状态和动作空间是高维连续时，使用Q-Table不现实。

通常做法是把Q-Table的更新问题变成一个函数拟合问题，相近的状态得到相近的输出动作。如下式，通过更新参数 θθ 使Q函数逼近最优Q值 
而深度神经网络可以自动提取复杂特征，因此，面对高维且连续的状态使用深度神经网络最合适不过了。

Q(s,a;θ)≈Q′(s,a)

DRL是将深度学习（DL）与强化学习（RL）结合，直接从高维原始数据学习控制策略。而DQN是DRL的其中一种算法，它要做的就是将卷积神经网络（CNN）和Q-Learning结合起来，CNN的输入是原始图像数据（作为状态State），输出则是每个动作Action对应的价值评估Value Function（Q值）。

## DL与RL结合的问题
DL需要大量带标签的样本进行监督学习；RL只有reward返回值，而且伴随着噪声，延迟（过了几十毫秒才返回），稀疏（很多State的reward是0）等问题；
DL的样本独立；RL前后state状态相关；
DL目标分布固定；RL的分布一直变化，比如你玩一个游戏，一个关卡和下一个关卡的状态分布是不同的，所以训练好了前一个关卡，下一个关卡又要重新训练；
过往的研究表明，使用非线性网络表示值函数时出现不稳定等问题。

##DQN解决问题方法
通过Q-Learning使用reward来构造标签（对应问题1）
通过experience replay（经验池）的方法来解决相关性及非静态分布问题（对应问题2、3）
使用一个CNN（MainNet）产生当前Q值，使用另外一个CNN（Target）产生Target Q值（对应问题4）

###构造标签
前面提到DQN中的CNN作用是对在高维且连续状态下的Q-Table做函数拟合，而对于函数优化问题，监督学习的一般方法是先确定Loss Function，然后求梯度，使用随机梯度下降等方法更新参数。DQN则基于Q-Learning来确定Loss Function。

Q-Learning 
有关RL的基础知识不再啰嗦，直接看Q-Learning的更新公式：

Q∗(s,a)=Q(s,a)+α(r+γmax_a′Q(s′,a′)−Q(s,a))

而DQN的Loss Function为:

L(θ)=E[(TargetQ−Q(s,a;θ))2]

其中 θ 是网络参数，目标为

TargetQ=r+γmax_a′Q(s′,a′;θ)

显然Loss Function是基于Q-Learning更新公式的第二项确定的，两个公式意义相同，都是使当前的Q值逼近Target Q值。

接下来，求 L(θ) 关于 θ 的梯度，使用SGD等方法更新网络参数 θ。

###经验池（experience replay）
经验池的功能主要是解决相关性及非静态分布问题。具体做法是把每个时间步agent与环境交互得到的转移样本 (st,at,rt,st+1)(st,at,rt,st+1) 储存到回放记忆单元，要训练时就随机拿出一些（minibatch）来训练。（其实就是将游戏的过程打成碎片存储，训练时随机抽取就避免了相关性问题）

###目标网络
在Nature 2015版本的DQN中提出了这个改进，使用另一个网络（这里称为TargetNet）产生Target Q值。具体地，Q(s,a;θi)Q(s,a;θi) 表示当前网络MainNet的输出，用来评估当前状态动作对的值函数；Q(s,a;θ−i)Q(s,a;θi−) 表示TargetNet的输出，代入上面求 TargetQTargetQ 值的公式中得到目标Q值。根据上面的Loss Function更新MainNet的参数，每经过N轮迭代，将MainNet的参数复制给TargetNet。

引入TargetNet后，再一段时间里目标Q值使保持不变的，一定程度降低了当前Q值和目标Q值的相关性，提高了算法稳定性。

##总结

DQN是第一个将深度学习模型与强化学习结合在一起从而成功地直接从高维的输入学习控制策略。

###创新点：

基于Q-Learning构造Loss Function（不算很新，过往使用线性和非线性函数拟合Q-Table时就是这样做）。
通过experience replay（经验池）解决相关性及非静态分布问题；
使用TargetNet解决稳定性问题。

###优点：

算法通用性，可玩不同游戏；
End-to-End 训练方式；
可生产大量样本供监督学习。

###缺点：

无法应用于连续动作控制；
只能处理只需短时记忆问题，无法处理需长时记忆问题（后续研究提出了使用LSTM等改进方法）；
CNN不一定收敛，需精良调参。
# Gazebo and Gym and Unity

### Gazebo-DQN

### Gym-DQN

![](IMG/gym_env.png)



### Unity