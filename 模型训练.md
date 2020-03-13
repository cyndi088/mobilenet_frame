# 模型训练

## 一.数据准备

### 1.使用labelImg工具标注图片样本；

### 2.图片样本保存于images目录中，xml格式的标注数据保存到merged_xml中；

### 3.运行train_test_split.py，将标注数据集分为了train、test、validation三部分，并存于annotations目录中；

### 4.运行xml_to_csv.py，将xml转为csv文件，并保存在data目录中；

### 5.生成tfrecords文件

```
python generate_tfrecord.py --csv_input=data/detection_train_labels.csv --output_path=data/detection_train.tfrecord  
```

```
python generate_tfrecord.py --csv_input=data/detection_validation_labels.csv --output_path=data/detection_validation.tfrecord  
```

```
python generate_tfrecord.py --csv_input=data/detection_test_labels.csv --output_path=data/detection_test.tfrecord  
```

## 训练

### 1.在data目录中创建标签分类配置文件(label_map.pbtxt)，需要检测几种目标，将创建几个id

```
item {
  id: 1 # id从1开始编号
  name: 'person'
}
item {
  id: 2
  name: 'investigator'
}
item {
  id: 3
  name: 'collector'
}
item {
  id: 4
  name: 'wolf'
}
```

### 2.配置模型管道配置文件(ssd*.config)

```
1.标注的总类别数；
2.训练集、验证集的tfrecord文件的路径；
3.label_map的路径
```

```
python train.py --logtostderr \
--pipeline_config_path=data/ssd_inception_v2_pets.config \
--train_dir=data
```

## TensorBoard监控

```
tensorboard --logdir==training:data
```

## 模型导出

```
python export_inference_graph.py \
--input_type image_tensor
--pipeline_config_path=data/ssd_inception_v2_pets.config \
--trained_checkpoint_prefix data/model.ckpt-30000 \
--output_directory data/exported_model_directory
# 工程的data目录下生成名为exported_model_directory文件夹
# frozen_inference_graph.pb就是我们以后将要使用的模型结果
```

## 获取测试图片

### 1.新建test_images目录

### 2.运行get_testImages.py，获取测试图片并存储到test_images文件夹目录下

## 批量保存测试结果

### 1.新建results目录

### 2.运行get_allTestResults.py，使用前面训练出的模型批量测试test_images文件夹中的图片并保存到results文件夹中
