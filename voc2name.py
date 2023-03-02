import os
import xml.etree.ElementTree as ET

#程序功能：批量修改VOC数据集中xml标签文件的标签名称
def changelabelname(inputpath):
    listdir = os.listdir(inputpath)
    for file in listdir:
        if file.endswith('xml'):
            file = os.path.join(inputpath,file)
            tree = ET.parse(file)
            root = tree.getroot()


            path = root.find('path')
            filename = os.path.basename(file)
            name1 = os.path.basename(path.text)

            if 'w' in filename:
                path.text = "Users\sinjunshe\Deskto\w"+name1
                tree.write(file, encoding='utf-8')
                print("end")

            #for object1 in root.findall('path'):

                # for sku in object1.findall('name'):           #查找需要修改的名称
                #     if (sku.text == "orange(Occluded)"):               #‘preName’为修改前的名称
                #         sku.text = "orange"               #‘TESTNAME’为修改后的名称
                #         tree.write(file,encoding='utf-8')     #写进原始的xml文件并避免原始xml中文字符乱码
                #         print("end")
                #     if (sku.text == "orange(Non-occluded)"):               #‘preName’为修改前的名称
                #         sku.text = "orange"               #‘TESTNAME’为修改后的名称
                #         tree.write(file,encoding='utf-8')     #写进原始的xml文件并避免原始xml中文字符乱码
                #         print("end")


        else:
            pass

if __name__ == '__main__':
    inputpath = 'finet/train/ann'  #此处替换为自己的路径
    changelabelname(inputpath)