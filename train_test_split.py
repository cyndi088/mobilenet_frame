import os
import random
import time
import shutil


xmlfilepath = '/home/merged_xml'
saveBasePath = './annotations'

trainval_percent = 0.9
train_percent = 0.85
total_xml = os.listdir(xmlfilepath)
num = len(total_xml)
ls = range(num)

# 训练集＋验证集共占90%
tv = int(num * trainval_percent)
# 训练集占90%中的85%,也就是76.5%
tr = int(tv * train_percent)
trainval = random.sample(ls, tv)
train = random.sample(trainval, tr)
print("train and val size", tv)
print("train size", tr)

start = time.time()

test_num = 0
val_num = 0
train_num = 0


def copy(dir, name):
    xml_path = os.path.join(os.getcwd(), 'annotations/{}'.format(dir))
    if not os.path.exists(xml_path):
        os.mkdir(xml_path)
    filePath = os.path.join(xmlfilepath, name)
    newfile = os.path.join(saveBasePath, os.path.join(dir, name))
    shutil.copyfile(filePath, newfile)


for i in ls:
    name = total_xml[i]
    if i in trainval:
        if i in train:
            directory = "train"
            train_num += 1
            copy(directory, name)
        else:
            directory = "validation"
            val_num += 1
            copy(directory, name)
    else:
        directory = "test"
        test_num += 1
        copy(directory, name)

end = time.time()
seconds = end - start
print("train total : " + str(train_num))
print("validation total : " + str(val_num))
print("test total : " + str(test_num))
total_num = train_num + val_num + test_num
print("total number : " + str(total_num))
print("Time taken : {0} seconds".format(seconds))
