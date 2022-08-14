- [MIRFlickr](https://press.liacs.nl/mirflickr/mirdownload.html)
  - 需要下载 mirflickr25k.zip 和 mirflickr25k_annotations_v080.zip

Flickr-25K 有 2,5000 张图，每张图有对应的 tags 和 annotation。
tags 可作为文本描述（text），其中至少出现在 20 张图片中的 tags 有 1386 个；
annotation 作为 label，一共 24 个。

最终将 image 处理成 VGG19 的 4096-D 特征、text 是 1386-D BoW 向量、label 是 24-D 0/1 向量。

- mirflickr/ 下是图像
- mirflickr/doc/ 下有 common_tags.txt，里面是上述的 1386 个 tags 和其对应的出现频数
- mirflickr/meta/tags/ 是每张图对应的处理过的 tags
- mirflickr25k_annotations_v080/ 下是各 annotations 的文件，表示每个类中含有的图像标号