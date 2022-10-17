'''
Description: Own dataset detection demo
Autor: Joy
Email: JoyZheng@human-horizons.com
LastEditors: Joy
Date: 2022-10-10 06:16:41
LastEditTime: 2022-10-14 11:41:43
'''
import asyncio
from argparse import ArgumentParser
import os
from pathlib import Path
from collections import Sequence
import mmcv
from mmcv import Config, DictAction
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root

from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)


def parse_args():
    parser = ArgumentParser()
    # parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    # parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='coco',
        choices=['coco', 'voc', 'citys', 'random'],
        help='Color palette used for visualization')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--async-test',
        action='store_true',
        help='whether to set async options for async inference.')
    # Joy
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
            'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg


def main(args):
    
    # get config
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # dataset
    valid_dataset = build_dataset(cfg.data.val)
    
    progress_bar = mmcv.ProgressBar(len(valid_dataset))
    for item in valid_dataset:
        # filename = os.path.join(args.output_dir,
        #                         Path(item['filename']).name
        #                         ) if args.output_dir is not None else None
        
        # exit(0)
        # test a single image
        img_path = item['img_metas'][0].data['filename']
        img_name = item['img_metas'][0].data['ori_filename']
        result = inference_detector(model, img_path)
        # show the results
        show_result_pyplot(
            model,
            img_path,
            result,
            palette=args.palette,
            score_thr=args.score_thr,
            out_file=os.path.join(args.output_dir, img_name)if args.output_dir is not None else None
        )
        progress_bar.update()




async def async_main(args):
    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    tasks = asyncio.create_task(async_inference_detector(model, args.img))
    result = await asyncio.gather(tasks)
    # show the results
    show_result_pyplot(
        model,
        args.img,
        result[0],
        palette=args.palette,
        score_thr=args.score_thr,
        out_file=args.out_file)


if __name__ == '__main__':
    args = parse_args()
    if args.async_test:
        asyncio.run(async_main(args))
    else:
        main(args)
