%matplotlib inline
import sys
import os
import time

cocoapi_pycocotools_PATH ="/home/steve_lin/cocoapi/PythonAPI/"
cv2_WRONG_PATH = '/opt/ros/kinetic/lib/python2.7/dist-packages'

if cocoapi_pycocotools_PATH not in sys.path:
    sys.path.append(cocoapi_pycocotools_PATH)

if cv2_WRONG_PATH in sys.path:
    sys.path.remove(cv2_WRONG_PATH)

import cython
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/home/steve_lin/coco'
dataType='val2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]

#split 80 cats into 4 sets, each of them include 20cats
S1 = list();S2 = list();S3 = list();S4 = list();
for i in range(0,len(nms),4):
    S1.append(nms[i])
    S2.append(nms[i])
    S3.append(nms[i])
    S4.append(nms[i])
# print('datadet S1: \n{}\n'.format(' '.join(S1)))
# print('datadet S2: \n{}\n'.format(' '.join(S2)))
# print('datadet S3: \n{}\n'.format(' '.join(S3)))
# print('datadet S4: \n{}\n'.format(' '.join(S4)))

#choose a dataset 
a = np.random.choice(S1)

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=[a]);
imgIds = coco.getImgIds(catIds=catIds);

b = int(np.random.choice(imgIds))
c = int(np.random.choice(imgIds))
while c == b:
    c = int(np.random.choice(imgIds))
# print('image1_id: '+str(b))
# print('image2_id: '+str(c))

imgIds1 = coco.getImgIds(imgIds = [b])
imgIds2 = coco.getImgIds(imgIds = [c])
img1 = coco.loadImgs(imgIds1[np.random.randint(0,len(imgIds1))])[0]
img2 = coco.loadImgs(imgIds2[np.random.randint(0,len(imgIds2))])[0]

#generate the image path 
str1 = ''
str2 = ''
ID1 = str(img1['id'])
ID2 = str(img2['id'])
for i in range(12-len(ID1)):
    str1 = str1+'0'
for i in range(12-len(ID2)):
    str2 = str2+'0'
val_dir1 = '/home/steve_lin/coco/val2017/'+str1+ID1+'.jpg'
train_dir1 = '/home/steve_lin/coco/train2017/'+str1+ID1+'.jpg'
val_dir2 = '/home/steve_lin/coco/val2017/'+str2+ID2+'.jpg'
train_dir2 = '/home/steve_lin/coco/train2017/'+str2+ID2+'.jpg'

#get annotations
annIds1 = coco.getAnnIds(imgIds=img1['id'], catIds=catIds, iscrowd=None)
anns1 = coco.loadAnns(annIds1)
annIds2 = coco.getAnnIds(imgIds=img2['id'], catIds=catIds, iscrowd=None)
anns2 = coco.loadAnns(annIds2)

for i in range(len(anns1)):
    print('id: '+str(anns1[i]['id']))
    print('bbox: ');print(anns1[i]['bbox']);print()                 #bbox_info
    print('segmentation: ');print(anns1[i]['segmentation']);print() #mask_info

for i in range(len(anns2)):
    print('id: '+str(anns2[i]['id']))
    print('bbox: ');print(anns2[i]['bbox']);print()                 #bbox_info
    print('segmentation: ');print(anns2[i]['segmentation']);print() #mask_info

## load and display image
##I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
## use url to load image
# if os.path.exists(val_dir1):
#     I = io.imread(val_dir1)
# else:
#     I = io.imread(train_dir1)
# plt.axis('off')
# plt.imshow(I)
# coco.showAnns(anns1)
## load and display instance annotations
