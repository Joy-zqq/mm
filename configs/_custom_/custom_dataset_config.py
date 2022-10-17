'''
Description: 
Autor: Joy
Email: JoyZheng@human-horizons.com
LastEditors: Joy
Date: 2022-09-27 11:27:02
LastEditTime: 2022-09-29 06:18:50
'''
# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=1),
#         mask_head=dict(num_classes=1)))
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)))

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('car',)
data = dict(
    train=dict(
        img_prefix='0_data/BDD100K/train_images/',
        classes=classes,
        ann_file='0_data/BDD100K/annotations/bdd100k_2d_train.json'),
    val=dict(
        img_prefix='0_data/BDD100K/valid_images/',
        classes=classes,
        ann_file='0_data/BDD100K/annotations/bdd100k_2d_valid.json'),
    # test=dict(
    #     img_prefix='balloon/val/',
    #     classes=classes,
    #     ann_file='balloon/val/annotation_coco.json')
    )

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = '0_checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
# https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth

