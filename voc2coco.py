import sys
import os
import json
import warnings
import numpy as np
import xml.etree.ElementTree as ET
import glob

START_BOUNDING_BOX_ID = 1
# 按照你给定的类别来生成你的 category_id
# COCO 默认 0 是背景类别
# CenterNet 里面类别是从0开始的，否则生成heatmap的时候报错
PRE_DEFINE_CATEGORIES = {"orange(occluded)": 1,"orange(Non-occluded)":2}
START_IMAGE_ID = 0


# If necessary, pre-define category and its id
#  PRE_DEFINE_CATEGORIES = {"aeroplane": 1, "bicycle": 2, "bird": 3, "boat": 4,
#  "bottle":5, "bus": 6, "car": 7, "cat": 8, "chair": 9,
#  "cow": 10, "diningtable": 11, "dog": 12, "horse": 13,
#  "motorbike": 14, "person": 15, "pottedplant": 16,
#  "sheep": 17, "sofa": 18, "train": 19, "tvmonitor": 20}


def get(root, name):
    vars = root.findall(name)
    return vars


def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


def get_filename_as_int(filename):
    try:
        filename = filename.replace("\\", "/")
        filename = os.path.splitext(os.path.basename(filename))[0]
        return int(filename)
    except:
        # raise ValueError("Filename %s is supposed to be an integer." % (filename))
        image_id = np.array([ord(char) % 10000 for char in filename], dtype=np.int32).sum()
        # print(image_id)
        return 0


def get_categories(xml_files):
    """Generate category name to id mapping from a list of xml files.

    Arguments:
        xml_files {list} -- A list of xml file paths.

    Returns:
        dict -- category name to id mapping.
    """
    classes_names = []
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall("object"):
            classes_names.append(member[0].text)
    classes_names = list(set(classes_names))
    classes_names.sort()
    return {name: i for i, name in enumerate(classes_names)}


def convert(xml_files, json_file):
    json_dict = {"images": [], "type": "instances", "annotations": [], "categories": []}
    if PRE_DEFINE_CATEGORIES is not None:
        categories = PRE_DEFINE_CATEGORIES
    else:
        categories = get_categories(xml_files)
    bnd_id = START_BOUNDING_BOX_ID
    image_id = START_IMAGE_ID
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = get(root, "path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))

        ## The filename must be a number
        # image_id = get_filename_as_int(filename)
        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)

        if ".jpg" not in filename or ".png" not in filename:
            filename = filename
            warnings.warn("filename's default suffix is jpg")

        images = {
            "file_name": filename,  # 图片名
            "height": height,
            "width": width,
            "id": image_id,  # 图片的ID编号（每张图片ID是唯一的）
        }
        json_dict["images"].append(images)
        ## Currently we do not support segmentation.
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        for obj in get(root, "object"):
            category = get_and_check(obj, "name", 1).text
            if category not in categories:
                new_id = len(categories)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            if 'w' in filename:
                domain = 1
            if 'w' not in filename:
                domain = 0
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,  # 对应的图片ID（与images中的ID对应）
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id, # 同一张图片可能对应多个 ann
                "ignore": 0,
                "segmentation": [],
                "domain": domain,
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1
        image_id += 1

    for cate, cid in categories.items():
        cat = {"supercategory": "none", "id": cid, "name": cate}
        json_dict["categories"].append(cat)

    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json.dump(json_dict, open(json_file, 'w'), indent=4)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(
    #     description="Convert Pascal VOC annotation to COCO format."
    # )
    # parser.add_argument("xml_dir", help="Directory path to xml files.", type=str)
    # parser.add_argument("json_file", help="Output COCO format json file.", type=str)
    # args = parser.parse_args()
    # args.xml_dir
    # args.json_file

    xml_dir = "finet/train/ann"
    json_file = "finet/train/train.json"  # output json
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))

    # If you want to do train/test split, you can pass a subset of xml files to convert function.
    print("Number of xml files: {}".format(len(xml_files)))
    convert(xml_files, json_file)
    print("Success: {}".format(json_file))

