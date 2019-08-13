import os
import random
import xml.etree.ElementTree as ET

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor',
           'plasticbag')

plasticbag_ann_path = './annotations'
plasticbag_img_path = './images'

voc_path = '../VOCdevkit/VOC2007'
voc_ann_path = os.path.join(voc_path, 'Annotations')
voc_img_path = os.path.join(voc_path, 'JPEGImages')
voc_sets_path = os.path.join(voc_path, 'ImageSets', 'Main')

voc_backup_path = '../VOCdevkit/VOC2007_backup'

def copy(dir_from, dir_to):
    """ Copy all files from dir_from to dir_to """
    cmd = 'cp'
    for filename in os.listdir(dir_from):
        cmd = cmd + ' ' + os.path.join(dir_from, filename)
    cmd = cmd + ' ' + dir_to
    os.system(cmd)

copy(plasticbag_ann_path, voc_ann_path)
copy(plasticbag_img_path, voc_img_path)

idxs = [os.path.splitext(img)[0] for img in os.listdir(plasticbag_img_path)]
# plasticVN images
idxs_vn = [idx for idx in idxs if idx.startswith('vn_')]
# OpenImagesV4 + ImageNet images
idxs = [idx for idx in idxs if not idx.startswith('vn_')]

# Shuffle all images before splitting into sets
random.shuffle(idxs_vn)
random.shuffle(idxs)

# Create train, val, test sets of plasticbag
## From OpenImagesV4 + ImageNet
train_num = int(len(idxs) / 10 * 8.5)
val_num = len(idxs) - train_num
train = idxs[:train_num]
val = idxs[train_num:]

## From plasticVN - my own dataset
train_num = int(len(idxs_vn) / 10 * 4)
val_num = int(len(idxs_vn) / 10 * 1)
test_num = len(idxs_vn) - train_num - val_num
train = train + idxs_vn[:train_num]
val = val + idxs_vn[train_num:train_num+val_num]
test = idxs_vn[train_num+val_num:]

# Integrate with the sets of Pascal VOC
def read_set(dir_set):
    return [line[0:-1] for line in open(dir_set)]

print('plasticbag train size', len(train))
print('plasticbag val size', len(val))
print('plasticbag test size', len(test))

train = train + read_set(os.path.join(voc_backup_path, 'ImageSets', 'Main', 'train.txt'))
val = val + read_set(os.path.join(voc_backup_path, 'ImageSets', 'Main', 'val.txt'))
test = test + read_set(os.path.join(voc_backup_path, 'ImageSets', 'Main', 'test.txt'))
trainval = train + val

# Rewrite the .txt files of the sets
def load_objects(idx):
    """ Load objects from .XML files """
    filename = os.path.join(voc_ann_path, idx + '.xml')
    tree = ET.parse(filename)
    objs = tree.findall('object')
    objs = [obj for obj in objs if obj.find('name').text in CLASSES]
    return [
        {
            'name': obj.find('name').text,
            'difficult': int(obj.find('difficult').text)
        } for obj in objs
    ]

def write_set(set, set_suffix):
    """ Rewrite .txt files of the set """
    """ e.g. train.txt, aeroplane_train.txt, ..., plasticbag_train.txt """
    objs = {}
    for idx in set:
        objs[idx] = load_objects(idx)

    w = open(os.path.join(voc_sets_path, set_suffix + '.txt'), 'w')
    for idx in set:
        w.write(idx + '\n')
    w.close()

    for cls_name in CLASSES[1:]: # ignore __background__
        w = open(os.path.join(voc_sets_path, cls_name + '_' + set_suffix + '.txt'), 'w')
        for idx in set:
            has = sum(1 for obj in objs[idx] if obj['name'] == cls_name)
            dif = sum(1 for obj in objs[idx] if obj['name'] == cls_name and obj['difficult'] == 1)
            if has == 0:
                w.write(idx + ' -1\n')
            else:
                if has == dif:
                    w.write(idx + '  0\n')
                else:
                    w.write(idx + '  1\n')
        w.close()

write_set(test, 'test')
write_set(val, 'val')
write_set(train, 'train')
write_set(trainval, 'trainval')