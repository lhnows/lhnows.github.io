
# SLAM3R：基于单目RGB视频的实时高精度密集场景重建

## 作者
刘宇征<sup>1</sup>\*  董诗言<sup>2</sup>\*† 王淑哲<sup>3</sup> 银莹达<sup>1</sup> 杨延朝<sup>2</sup>† 樊清楠<sup>4</sup> 陈宝权<sup>1</sup>†  
<sup>1</sup>北京大学 <sup>2</sup>香港大学 <sup>3</sup>阿尔托大学 <sup>4</sup>VIVO

---

## 摘要
本文提出SLAM3R系统，通过单目RGB视频实现实时、高质量的密集三维场景重建。该系统通过前馈神经网络无缝整合局部三维重建与全局坐标系配准，形成端到端解决方案。输入视频经滑动窗口机制分割为重叠片段，SLAM3R直接从每个窗口的RGB图像中回归三维点云，并通过渐进对齐和形变生成全局一致的场景模型，无需显式求解相机参数。在多个数据集上，SLAM3R在保持20+ FPS实时性能的同时，实现了精度与完整度的当前最优重建效果。代码已开源：[https://github.com/PKU-VCL-3DV/SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R)

---

### 图1. SLAM3R系统架构
![](https://code-liuhao.oss-cn-hangzhou.aliyuncs.com/papers/2412.09401v3/_page_0_Figure_8.jpeg)

---

## 1. 引言
传统三维重建依赖多阶段流程：通过稀疏SLAM/SfM估计相机参数，再通过MVS填充场景细节。这类方法虽精度高，但需要离线处理。近期单目SLAM系统虽尝试实时重建，但普遍存在精度不足或效率低下问题。例如NICER-SLAM仅能实现<1 FPS速度。

---

### 图2. 系统整体流程
![](https://code-liuhao.oss-cn-hangzhou.aliyuncs.com/papers/2412.09401v3/_page_3_Figure_0.jpeg)

---

## 3. 方法
### 3.1 局部重建（Inner-Window）
I2P网络采用多分支Vision Transformer架构，包含共享图像编码器和分离解码器：
$$F_i^{(T × d)} = E_{img}(I_i^{(H × W × 3)})$$
关键帧选择窗口中央图像，通过多视图交叉注意力融合支持帧信息，最终通过线性头回归三维点云：
$$(\hat{X}_i^{(H × W × 3)}, \hat{C}_i^{(H × W × 1)}) = \mathbf{H}(G_i^{(T × d)})$$

---

### 图3. 解码器块结构
![](https://code-liuhao.oss-cn-hangzhou.aliyuncs.com/papers/2412.09401v3/_page_3_Figure_4.jpeg)

---

## 4. 实验结果
### 表1. 7-Scenes数据集对比
| 方法               | 准确度/完整度(cm) | FPS  |
|--------------------|-------------------|------|
| SLAM3R (Ours)      | 2.13/2.34         | ∼25  |
| Spann3R [61]       | 3.42/2.41         | >50  |
| DUSt3R [64]        | 2.19/3.24         | <1   |

---

### 图4. 重建效果对比
![](https://code-liuhao.oss-cn-hangzhou.aliyuncs.com/papers/2412.09401v3/_page_5_Figure_6.jpeg)

---

## 5. 结论
SLAM3R通过双层级神经网络架构，在无需显式求解相机参数的情况下，实现了20+ FPS的实时重建。实验表明其在精度（2.13cm）和完整度（2.34cm）方面优于现有方法，为单目RGB视频的密集重建提供了高效解决方案。