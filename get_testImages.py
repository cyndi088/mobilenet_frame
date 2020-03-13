from PIL import Image
import os.path


annotations_test_dir = "/home/yq2/PycharmProjects/object_detection_2019_v2.2/annotations/test/"
Image_dir = "/home/images"
test_images_dir = "/home/yq2/PycharmProjects/object_detection_2019_v2.2/test_images"
i = 0
for xmlfile in os.listdir(annotations_test_dir):
    filepath, tempfilename = os.path.split(xmlfile)
    shotname, extension = os.path.splitext(tempfilename)
    xmlname = shotname
    for jpgfile in os.listdir(Image_dir):
        filepath, tempfilename = os.path.split(jpgfile)
        jpgname, extension = os.path.splitext(tempfilename)
        if jpgname == xmlname:
            img = Image.open(Image_dir + "/" + jpgname + ".jpg")
            img.save(os.path.join(test_images_dir, os.path.basename(jpgfile)))
            print(jpgname)
            i += 1
print(i)
