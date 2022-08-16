## 原数据
[NUS-WIDE](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)
- 多标签（multi-label）数据集，26,9648 个样本、81 个类。
- 文件结构
  - Groundtruth，label，解压后有 AllLabels/ 和 TrainTestLabels/ 两个目录
  - Tags，可以做 text 模态数据，下载得 NUS_WID_Tags.zip
  - Concept List，解压得 Concepts81.txt，81 个类的类名
  - Image List，其中 Imagelist.txt 指明每个样本对应的 image
  - Image Urls，给出 image 数据的下载链

## 问题
用原数据进行清洗，最后剩余 19,0421 个数据。与 DGCPN 文中所述的 186577 条数据量不符。
### 解决方案
直接使用 DCMH 中所提供的数据集：
- NUS-WIDE (top-10 concept):
  - link: https://pan.baidu.com/s/1GFljcAtWDQFDVhgx6Jv_nQ
  - password: ml4y
读取并使用VGG19提取image特征
