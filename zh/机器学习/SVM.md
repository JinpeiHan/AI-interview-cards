# SVM

Q1.***什么是SVM***？

- **SVM（支持向量机）是一种监督学习方法，核心思想是在特征空间里找一个超平面，把两类数据分开，并让两类之间的间隔（margin）最大。**
- 它只关注最难区分、离边界最近的点（支持向量），而不是所有样本。

Q2:***SVM中的超平面是什么***？

- **超平面就是 SVM 的决策边界：落在它两侧的点被分到不同类别。它的维度比特征空间少一维——2 个特征时是一条直线，3 个特征时是一个平面。**
- 能把数据分开的超平面有很多，SVM 选的是间隔最大的那一个。

![https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.png](https://miro.medium.com/max/600/0*0o8xIA4k3gXUDCFU.png)

Q3:***SVM 中的支持向量是什么***？

- **支持向量是离超平面最近、正好落在间隔边界上（或间隔内）的样本点。它们决定了超平面的位置：去掉它们，超平面会变；去掉其他点，超平面不变。**
- 预测时也只用到支持向量，所以 SVM 是稀疏模型。

![https://miro.medium.com/max/1400/0*ecA4Ls8kBYSM5nza.jpg](https://miro.medium.com/max/1400/0*ecA4Ls8kBYSM5nza.jpg)

Q4:***什么是hard-margin和soft-margin SVM***？

- **硬间隔 SVM 要求所有点都被正确分开且不进入间隔区，只适用于线性可分数据；软间隔 SVM 允许少数点落入间隔内甚至分错，但要付出惩罚，适用于有噪声、不完全线性可分的数据。**

Q5:***你知道哪些类型的SVM内核***？

- **常用的有线性核、多项式核、高斯（RBF）核，此外还有 Sigmoid 核。**

Q6:***为什么使用核技巧***？

- **数据线性不可分时，需要把它映射到高维空间才能线性分开，但直接在高维算代价太大。核技巧让我们在原始低维空间里直接算出高维空间的内积，从而得到高维的分类效果，却不需要真的去做映射。**
- SVM 的对偶形式只依赖样本间的内积，所以可以直接把内积换成核函数。

Q7:***SVM有哪些应用***？

- **SVM 主要用于分类，典型应用有人脸检测、文本分类、图像分类、生物信息学（蛋白质、癌症分类）、手写识别等，尤其适合样本少、维度高的任务。**

Q8:***说出SVM的一些优点***？

- **SVM 是凸优化问题，能保证得到全局最优解；借助核技巧可以处理非线性问题；只依赖支持向量，在小样本、高维数据上泛化好。**
- 软间隔同时适用于线性可分和不可分数据。

Q9:***对于`N`维度数据集，支持向量的最小可能数量是多少？(假设数据是完全线性可分的)***

- **最少是 2 个（每类一个），与数据维度和样本数量无关。**

Q10:***当 SVM 中没有明确的超平面时会发生什么？***

- **如果原始空间里找不到能分开数据的超平面，就把数据映射到更高维的空间，在那里再找线性超平面；这个过程通过核函数完成，称为核化。**

![https://66.media.tumblr.com/9bffea56372d28d2a30f80557451e824/tumblr_inline_o9aabehtqP1u37g00_540.png](https://66.media.tumblr.com/9bffea56372d28d2a30f80557451e824/tumblr_inline_o9aabehtqP1u37g00_540.png)

Q11:***SVM中的hinge loss是什么？***

- **Hinge loss（合页损失）是软间隔 SVM 使用的损失：样本被正确分类且离边界足够远（函数间隔 ≥ 1）时损失为 0，否则损失随距离线性增加。所以即使分类正确，但离边界太近也会被惩罚。**
- 软间隔 SVM 等价于"hinge loss + L2 正则"：正则项让 $\|w\|$ 变小，即间隔变大。
- 核心公式：$\max(0,\ 1-y_i f(x_i))$

![https://app.yinxiang.com/shard/s24/res/8d14cae8-d22f-499a-aeb8-7a8e535fc62b.gif](https://app.yinxiang.com/shard/s24/res/8d14cae8-d22f-499a-aeb8-7a8e535fc62b.gif)

![https://miro.medium.com/max/575/1*PGqpYm7o5GCbDXxXErr2JA.png](https://miro.medium.com/max/575/1*PGqpYm7o5GCbDXxXErr2JA.png)

引用：[https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/](https://programmathically.com/understanding-hinge-loss-and-the-svm-cost-function/)

Q12：***超参数C在SVM中的作用***

- **C 是软间隔 SVM 中错分样本的惩罚系数，用来权衡"间隔尽量大"和"分错尽量少"。C 越大越不容忍错误，C 越小越容忍错误。**

Q13: ***讲述设计SVM分类器时常用的超参数***

- **最重要的是惩罚系数 C 和 RBF 核的 gamma，此外还有核函数类型（以及多项式核的次数）。**
- C：越大越不容忍错误，易过拟合；越小越容易欠拟合。
- gamma：决定单个样本影响范围，越大影响范围越小、边界越弯曲、支持向量往往越多，易过拟合；越小边界越平滑。

Q14:***比较KNN算法和SVM算法***

- **KNN 没有训练过程，预测时直接用全部训练样本算距离；SVM 需要训练出超平面，预测时只用支持向量。**
- KNN 是非稀疏模型，预测慢、存储大；SVM 是稀疏模型。
- KNN 主要只有参数 k；SVM 要调 C、核函数及其参数。

Q15:***使用SVM做分类和回归有什么区别？***

- **SVM 分类是让离超平面最近的点尽量远，即把两类分得最开；SVR 回归是让所有样本尽量落在超平面附近的一条"管道"里，使样本到超平面的偏差尽量小。**

引用：[https://www.cnblogs.com/belter/p/8975606.html](https://www.cnblogs.com/belter/p/8975606.html)

引用：https://www.quora.com/What-is-the-main-difference-between-a-SVM-and-SVR

Q16:***什么是凸包？***

- **一组点的凸包是能包住所有点的最小凸多边形（没有向内凹的角）。**
- 与 SVM 的关系：线性可分时，最大间隔超平面就是两类凸包之间最近两点连线的垂直平分面。下图左边为凸多边形，右边为非凸多边形。

![https://miro.medium.com/max/560/1*xuqyWJ49_oNxrXGE6TehVw.png](https://miro.medium.com/max/560/1*xuqyWJ49_oNxrXGE6TehVw.png)

Q17：***如何使用SVM进行异常检测？***

- **用 One-class SVM：只用正常数据训练，学出一个包住正常数据的边界（超平面或超球面），新样本落在边界内判为正常，落在外面判为异常。**

Q18: ***比较SVM和logistic回归***

- **两者都是线性分类模型，主要区别在损失函数：LR 用交叉熵（对数损失），考虑所有样本；SVM 用 hinge loss，只由边界附近的支持向量决定。**
- 异常值：SVM 只看边界附近的点，对远处异常点不敏感；LR 所有点都有贡献，更容易受影响。
- 非线性：SVM 靠核技巧；LR 主要靠特征工程（交叉特征、离散化）。
- 效率：海量数据下 LR 更快，核 SVM 很慢。

Q19：***解释什么是核技巧***

- **核技巧就是找一个核函数 $K$，让它在原始低维空间里算出的值恰好等于映射到高维空间后的内积。这样不用显式算高维映射 $\phi(x)$，就能得到高维空间中的分类效果。**
- 之所以只需要内积，是因为 SVM 的对偶问题和决策函数里只出现样本间的内积。
- 核心公式：$K(x,y)=\langle\phi(x),\phi(y)\rangle$

![https://app.yinxiang.com/shard/s24/res/f3d68831-c5ee-4e9c-8737-49753082af57.gif](https://app.yinxiang.com/shard/s24/res/f3d68831-c5ee-4e9c-8737-49753082af57.gif)

![https://app.yinxiang.com/shard/s24/res/d8e584b8-ff97-4155-8df7-b0e131e2f32a.gif](https://app.yinxiang.com/shard/s24/res/d8e584b8-ff97-4155-8df7-b0e131e2f32a.gif)

![https://app.yinxiang.com/shard/s24/res/44a1a853-026a-40fe-8752-da1d7302a9de.gif](https://app.yinxiang.com/shard/s24/res/44a1a853-026a-40fe-8752-da1d7302a9de.gif)

![https://app.yinxiang.com/shard/s24/res/8be1a479-ee3c-45c8-aac4-aa5f44f02ed6.gif](https://app.yinxiang.com/shard/s24/res/8be1a479-ee3c-45c8-aac4-aa5f44f02ed6.gif)

引用：[https://www.fanyeong.com/2017/11/13/the-kernel-trick/](https://www.fanyeong.com/2017/11/13/the-kernel-trick/)

Q20:***比较随机森林和SVM，什么情况下优先使用随机森林（或svm）？***

- **小样本、高维、异常值少的数据优先用 SVM；数据量较大、特征类型混杂、有缺失值时优先用随机森林，它通常更鲁棒、调参更省事。**
- SVM 没有处理缺失值的机制，而决策树可以处理；SVM 依赖特征空间中的距离，缺失特征影响很大。

Q21:***比较svm和deep learning***

- **SVM 是在固定特征上找最大间隔的边界，凸优化、有全局最优解、理论清晰；深度学习通过多层网络自动学习分层特征表示，表达能力强但需要大量数据，且优化非凸。**
- 可解释性：SVM 有完整的数学推导支撑，深度网络更像黑箱。
- 优化：神经网络用梯度下降，对初始化敏感，可能陷入局部最优；SVM 无论初始值如何都收敛到全局最优。
- 多分类：神经网络一个模型即可；SVM 需要组合多个二分类器。

Q22:***哪些情况不适合用SVM?***

- **样本量很大（核 SVM 训练太慢）、缺失值多、多分类，以及特征维度远大于样本数的情况，SVM 都不太合适。**

Q23:***解释什么是支持向量？***

- **支持向量是离分隔超平面最近的样本点，只有它们决定了超平面的位置：移除它们超平面会改变，移除其他点则不变。预测时也只用到支持向量。**

Q24:***提供软间隔支持向量机 (SVM)的直观解释***

- **现实数据有噪声和离群点，硬间隔可能无解，或者为了迁就个别离群点得到一个很窄的间隔。软间隔允许少数点越界（落入间隔甚至分错），但每次越界都要付出代价，本质是在"间隔尽量宽"和"分错尽量少"之间做权衡。**
- 每个样本有一个松弛变量 $\xi_i$ 表示越界程度：为 0 表示在间隔外，0~1 表示在间隔内但分对，大于 1 表示分错。
- C 越大越接近硬间隔（间隔窄、易过拟合），C 越小越宽容（间隔宽、可能欠拟合）。
- 它等价于"hinge loss + L2 正则"（见 Q11）；在对偶问题中只体现为乘子有上界 $0\le\alpha_i\le C$。
- 核心公式：$\min\ \frac{1}{2}\|w\|^2 + C\sum_i\xi_i$，约束 $y_i(w^Tx_i+b)\ge 1-\xi_i,\ \xi_i\ge 0$

Q25:***解释SVM中的凸优化问题***

- **凸优化要求目标函数是凸函数、不等式约束是凸函数、等式约束是仿射函数，这样局部最优就是全局最优。SVM 的原问题是最小化 $\frac{1}{2}\|w\|^2$（凸的二次函数），约束是关于 $w,b$ 的线性不等式，所以是一个凸二次规划问题。**
- 实际求解时用拉格朗日法转成对偶问题，再求解对偶问题。

![https://app.yinxiang.com/shard/s24/res/bbd4d0b0-071c-4661-bf9a-ec42d7014c74.gif](https://app.yinxiang.com/shard/s24/res/bbd4d0b0-071c-4661-bf9a-ec42d7014c74.gif)

![https://app.yinxiang.com/shard/s24/res/cb08997e-1d24-4cf3-86f8-a62e813f3df4.gif](https://app.yinxiang.com/shard/s24/res/cb08997e-1d24-4cf3-86f8-a62e813f3df4.gif)

![https://app.yinxiang.com/shard/s24/res/bd9293e2-da82-46ef-9835-d2d11c982742.gif](https://app.yinxiang.com/shard/s24/res/bd9293e2-da82-46ef-9835-d2d11c982742.gif)

![https://app.yinxiang.com/shard/s24/res/aa4102c0-7780-4825-87d4-2089b018264f.gif](https://app.yinxiang.com/shard/s24/res/aa4102c0-7780-4825-87d4-2089b018264f.gif)

![https://app.yinxiang.com/shard/s24/res/4a63af41-25e5-41ff-a5eb-0be2213030a9.gif](https://app.yinxiang.com/shard/s24/res/4a63af41-25e5-41ff-a5eb-0be2213030a9.gif)

![https://app.yinxiang.com/shard/s24/res/6c214eb0-6937-4e6f-b1e0-96a12199fc25.gif](https://app.yinxiang.com/shard/s24/res/6c214eb0-6937-4e6f-b1e0-96a12199fc25.gif)

Q26:***什么是多项式内核？***

- **多项式核用多项式函数把数据隐式映射到高维空间，让原本线性不可分的数据变得线性可分。**
- 属于全局核：相距很远的点也会影响核值。
- 次数 p 越大，映射维度越高、模型越复杂，易过拟合；p 太小则可能欠拟合。
- 核心公式：$K(x,y)=(\gamma\, x^Ty+r)^p$

![https://app.yinxiang.com/shard/s24/res/ed13238c-98af-4a65-a531-beee76ef7da3.gif](https://app.yinxiang.com/shard/s24/res/ed13238c-98af-4a65-a531-beee76ef7da3.gif)

引用：https://www.pycodemates.com/2022/10/svm-kernels-polynomial-kernel.html

Q27***:什么是高斯（RBF）核函数?***

- **高斯核根据两个样本的距离计算相似度，距离越近值越接近 1，越远越接近 0；它相当于把数据映射到无穷维空间。**
- 是局部核，只有附近的点相互影响明显。
- 参数少、在大小样本上表现都不错，**不知道选什么核时一般优先用 RBF 核**。
- 核心公式：$K(x,y)=\exp\left(-\gamma\|x-y\|^2\right)$，其中 $\gamma=\frac{1}{2\sigma^2}$

![https://app.yinxiang.com/shard/s24/res/19dcaf08-349c-44db-9b7e-6c45d59c3758.gif](https://app.yinxiang.com/shard/s24/res/19dcaf08-349c-44db-9b7e-6c45d59c3758.gif)

Q28:***决策边界和超平面有什么区别***?

- **超平面是一个几何概念，指比所在空间低一维的子空间（二维空间中是直线，三维空间中是平面）；决策边界是一个功能概念，指把不同类别分开的那条边界。**
- 在线性 SVM 中，决策边界正好是一个超平面；用了核函数后，决策边界在原始空间里通常是曲面。

Q29:***冗余数据会怎样影响SVM的性能***？

- **冗余或无关特征会降低 SVM 的分类性能：它们在映射到高维时引入额外噪声，干扰距离和内积的计算，从而影响分类面的选择。所以使用 SVM 前通常要做特征选择。**

引用：https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4709852/

Q30:***阐述拉格朗日方法在SVM算法中的作用***

- **拉格朗日乘子法把 SVM 带约束的凸二次规划问题转成拉格朗日函数，再利用强对偶性（满足 Slater 条件）转成对偶问题求解，最后通过 KKT 条件从乘子还原出 $w$ 和 $b$。**

Q31:***解释对偶问题在SVM中的作用***

- **对偶问题消去了 $w$ 和 $b$，把优化变成只关于拉格朗日乘子的问题，更容易求解；更重要的是，对偶形式中只出现样本间的内积，因此可以自然地引入核函数，处理非线性分类。**

Q32:***超参数C的值如何影响 SVM***?

- **C 越大，对分错样本的惩罚越重，模型尽量不放过任何离群点，间隔变窄、边界更复杂，容易过拟合；C 越小，越容忍离群点，间隔更宽、模型更简单，但可能欠拟合。**
- scikit-learn 中 C 默认是 1。

Q33:***超参数gamma的值如何影响 SVM***?

- **gamma 控制 RBF 核中单个样本的影响范围。gamma 越小，每个样本影响范围越大，决策边界越平滑；gamma 越大，影响范围越小，边界越弯曲，支持向量往往越多，容易过拟合。**
- scikit-learn 中默认 `gamma='scale'`，即 `1/(n_features * X.var())`（旧版本默认 `1/n_features`）。

Q34:***支持向量数与SVM分类器性能之间是否存在关系***?

- **有关系：支持向量越多，模型通常越复杂，越容易过拟合，预测也越慢；支持向量很少则模型简单，可能欠拟合。**

Q35:***什么是结构化支持向量机***？

- **结构化 SVM 是 SVM 的推广：普通 SVM 输出单个标签或数值，结构化 SVM 可以输出有结构的对象，比如序列、树或集合。**

Q36:***解释SMO算法的思想***；

- **SMO 是求解 SVM 对偶问题的高效算法：每次只挑两个拉格朗日乘子来优化，固定其他乘子，这个二变量子问题有解析解，反复迭代直到所有乘子都满足 KKT 条件。**
- 为什么选两个：对偶问题有等式约束 $\sum_i\alpha_i y_i=0$，只改一个乘子会破坏约束。
- 每一步都让目标函数下降，且子问题求解不需要数值优化，所以很快。

Q37:***说出使用支持向量机与逻辑回归进行分类时，分别的优势***

- **SVM 的优势是只关心边界附近的点，对异常点不敏感，并且能用核技巧高效处理非线性问题；逻辑回归的优势是训练快、适合大规模数据，还能直接输出概率。**

Q38***:什么情况下使用SVM，什么情况下使用逻辑回归***？

- **数据量大时用逻辑回归；数据量中小、异常点较多、或有非线性结构时用 SVM（带核）。**
- 特征维度远大于样本数时，数据往往已经线性可分，用逻辑回归或线性 SVM。
- 特征少、样本量适中时，用 RBF 核 SVM。
- 样本量很大时核 SVM 效率不高，用逻辑回归或线性 SVM。

Q39:***SVM中的松弛变量是什么***？

- **松弛变量 $\xi_i$ 是软间隔 SVM 给每个样本引入的"容忍度"，表示该样本违反间隔约束的程度。硬间隔要求函数间隔至少为 1，软间隔放宽为至少 $1-\xi_i$，以此容忍噪声和离群点。**
- $\xi_i$ 越大越界越严重，大于 1 表示被分错；目标函数中用 $C\sum_i\xi_i$ 惩罚。

Q40:***SVM中参数C和gamma的含义***

- **C 是惩罚因子，控制对分错样本的容忍度：C 越大越关注被误分类的点；gamma 是 RBF 核的参数，决定决策边界的弯曲程度：gamma 越大边界越弯曲，越小越平滑。**

引用：[https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine](https://www.quora.com/What-are-C-and-gamma-with-regards-to-a-support-vector-machine)

Q41:***您将如何使用SVM处理非线性可分数据的分类***

- **两种手段结合使用：用软间隔（松弛变量 + 惩罚因子 C）容忍少量分不开的点；用核函数把数据隐式映射到高维空间，使其在高维线性可分。**

Q42:***能直观地解释一下PAC学习理论吗？***

- **PAC（Probably Approximately Correct，可能近似正确）理论认为：由于只有有限样本、不知道真实分布，要求学到误差为 0 的模型不现实，只能要求算法以较大概率学到一个误差较小的模型。**
- PAC 可学习：算法能在多项式时间内、用合理数量的样本，以至少 $1-\delta$ 的概率学到误差不超过 $\epsilon$ 的假设。
- 它说明了需要多少样本：假设空间越大、要求的误差越小、置信度越高，所需样本就越多。

Q43:***支持向量机可以用于异常值检测吗？***

- **可以，用 One-class SVM。它只用正常样本训练，寻找一个超平面，把数据和原点之间的间隔最大化，相当于用一个"圈"圈住正常数据；新样本在圈内是正常，在圈外是异常。**

Q44:***SVM是否会遭受维度灾难?***

- **会部分受影响。核技巧让 SVM 不用在高维空间里显式计算，避免了计算量随维度爆炸；但维度高时数据变稀疏，模型仍可能学到噪声，过拟合风险依然存在。**

Q45:***什么是Mercer 定理，它与 SVM 有什么关系?***

- **Mercer 定理给出了一个函数能作为合法核函数的条件：对任意样本集，它对应的核矩阵都是对称半正定的。**
- 意义：判断一个函数是否是合法核，不需要找出具体的映射 $\phi$，只需检查核矩阵是否对称半正定。

Q46:***使用SVM时，如何优化RBF 内核，以适应大量训练样本？***

- **可以用随机特征（Random Fourier Features）：构造一个随机的显式映射，把数据映射到一个有限维空间，使该空间中的内积近似 RBF 核值，然后在上面训练线性 SVM。这样计算和存储随样本数线性增长，可扩展到大数据。**
- 该方法来自 NIPS 2007 的《Random Features for Large-Scale Kernel Machines》，映射通过蒙特卡洛采样构造。

引用：[https://maelfabien.github.io/machinelearning/largescale/](https://maelfabien.github.io/machinelearning/largescale/)

[https://www.zhihu.com/question/19591450](https://www.zhihu.com/question/19591450)

Q47:***为什么SVM现在不流行了？***

- **现在是大数据时代，样本量动辄百万级，核 SVM 的训练复杂度大约是样本数的平方到立方，扩展性差；而深度学习在大数据、图像、文本等任务上效果更好，还能自动学特征，所以 SVM 热度下降。**

Q48:***SVM什么时候表现不好？***

- **样本量非常大、核映射维度很高时计算量过大；特征维度远大于样本数时效果一般；数据中有大量缺失值时也表现不好。**

Q49:***SVM如何选择内核?***

- **特征维度高或样本量大时，用线性核（或逻辑回归）；特征维度低、样本量适中时，用高斯（RBF）核。**
- 特征维度高时，数据通常已接近线性可分，没必要再映射。
- 样本量大时，高斯核要两两计算样本内积，开销远大于线性核，可以手动加些特征后用线性模型。
