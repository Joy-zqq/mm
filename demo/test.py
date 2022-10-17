'''
Description: 
Autor: Joy
Email: JoyZheng@human-horizons.com
LastEditors: Joy
Date: 2022-09-26 08:22:21
LastEditTime: 2022-10-11 08:42:40
'''
from genericpath import exists
from mmdet.apis import init_detector, inference_detector


'''1'''
import json

config_idx = 10000
# config_folder = 'train'
config_folder = 'valid'
jsonFile = '0_data/BDD100K/annotations/bdd100k_2d_'+config_folder+'.json'
newJsonFile = '0_data/OwnBDD/bdd100k_2d_'+config_folder+'_own.json'
current_idx = 0
with open(jsonFile) as json_file:
	data = json.load(json_file)
	annotations = data['annotations'][0:config_idx]
	current_idx = data['annotations'][config_idx]['image_id']
	
	images = data['images'][0:current_idx+1]
	categories = data['categories']

	newJson = {'images':images, 'annotations': annotations, 'categories': categories}

	with open(newJsonFile, 'w') as new_json_file:
		new_json_file.write(str(newJson).replace('\'', '"').replace('True', '"True"').replace('False', '"False"'))

print('images len:'+str(current_idx+1))
# [{'id': 1, 'name': 'traffic light'}, 
# {'id': 2, 'name': 'traffic sign'}, 
# {'id': 3, 'name': 'car'}, 
# {'id': 4, 'name': 'pedestrian'}, 
# {'id': 5, 'name': 'bus'}, 
# {'id': 6, 'name': 'truck'}, 
# {'id': 7, 'name': 'rider'}, 
# {'id': 8, 'name': 'bicycle'}, 
# {'id': 9, 'name': 'motorcycle'},
# {'id': 10, 'name': 'train'}, 
# {'id': 11, 'name': 'other vehicle'}, 
# {'id': 12, 'name': 'other person'}, 
# {'id': 13, 'name': 'trailer'}]


# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# # 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# # 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
# checkpoint_file = '0_checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# device = 'cuda:7'
# # 初始化检测器
# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# inference_detector(model, 'demo/demo.jpg')


'''2'''
# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
# _base_ = '../configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# # 我们需要对头中的类别数量进行修改来匹配数据集的标注
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1),
#         mask_head=dict(num_classes=1)))

# # 修改数据集相关设置
# dataset_type = 'CocoDataset'
# classes = ('bg','bus',)
# data = dict(
#     train=dict(
#         img_prefix='0_data/BDD100K/train_images/',
#         classes=classes,
#         ann_file='0_data/BDD100K/annotation/bdd100k_2d_train.json'),
#     val=dict(
#         img_prefix='0_data/BDD100K/valid_images/',
#         classes=classes,
#         ann_file='0_data/BDD100K/annotation/bdd100k_2d_valid.json'),
#     test=dict(
#         img_prefix='0_data/BDD100K/valid_images/',
#         classes=classes,
#         ann_file='0_data/BDD100K/annotation/bdd100k_2d_valid.json')
#     )

# # 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
# load_from = '0_checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# # load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'



'''3'''
import mmcv

# # Specify the path to model config and checkpoint file
# config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
# checkpoint_file = '0_checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# # build the model from a config file and a checkpoint file
# model = init_detector(config_file, checkpoint_file, device='cuda:7')

# # test a single image and show the results
# img = 'demo/demo.jpg'  # or img = mmcv.imread(img), which will only load it once
# result = inference_detector(model, img)
# # visualize the results in a new window
# model.show_result(img, result)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='0_work_dir/result.jpg')

# # test a video and show the results
# video = mmcv.VideoReader('demo/demo.mp4')
# for frame in video:
#     result = inference_detector(model, frame)
#     model.show_result(frame, result, wait_time=1)