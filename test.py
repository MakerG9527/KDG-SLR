import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import clip
import torch.nn as nn
from datasets import Action_DATASETS
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
import pprint
import numpy
from modules.Visual_Prompt import visual_prompt
from modules.Text_Prompt import *
from utils.Augmentation import get_augmentation
import torch
from modules.mamba import MambaTextEncoder


def remove_dataparallel_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(epoch, val_loader, classes, device, model_image, fusion_model, config, model_text):
    model_image.eval()
    fusion_model.eval()
    model_text.eval()
    num = 0
    corr_1 = 0
    corr_5 = 0

    with torch.no_grad():
        classes, num_text_aug, text_dict = mamba_text_prompt_with_descriptions(val_loader.dataset)
        # 修复：从字典中提取文本描述列表
        if isinstance(text_dict, dict):
            # 如果是字典，提取所有值组成列表
            text_descriptions = [text_dict[i] for i in sorted(text_dict.keys())]
        else:
            # 如果已经是列表格式
            text_descriptions = text_dict

        # 修复：分批处理文本特征提取以避免内存不足
        text_features_list = []
        batch_size = 32  # 根据GPU内存调整批次大小

        for i in range(0, len(text_descriptions), batch_size):
            batch_texts = text_descriptions[i:i + batch_size]
            # 使用DataParallel包装的模型处理文本批次
            batch_features = model_text.module(batch_texts) if hasattr(model_text, 'module') else model_text(
                batch_texts)
            text_features_list.append(batch_features)

        # 合并所有批次的特征
        text_features = torch.cat(text_features_list, dim=0)
        text_features = text_features.half()

        for i, (image, label) in enumerate(tqdm(val_loader)):
            image = image.view((-1, config.data.num_segments, 3) + image.size()[-2:])
            b, t, c, h, w = image.size()
            image = image.to(device).view(-1, c, h, w)
            label = label.to(device)

            # 使用 ImageCLIP 包装器来编码图像
            image_embedding = model_image(image)
            image_embedding = image_embedding.view(b, t, -1)
            image_embedding = fusion_model(image_embedding)

            image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 获取 logit_scale
            logit_scale = model_image.module.model.logit_scale.exp()
            logits = logit_scale * image_embedding @ text_features.t()

            logits = logits.float()
            # 修复：label已经是在device上的张量，不需要再次转换
            ground_truth = label

            acc1, acc5 = accuracy(logits.cpu(), ground_truth.cpu(), topk=(1, 5))
            corr_1 += acc1.item()
            corr_5 += acc5.item()
            num += 1

    top1 = corr_1 / num
    top5 = corr_5 / num

    print('Top-1 Accuracy: {:.5f}%'.format(top1))
    print('Top-5 Accuracy: {:.5f}%'.format(top5))

    return top1


def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                       T=config.data.num_segments,
                                       dropout=config.network.drop_out, emb_dropout=config.network.emb_dropout,
                                       pretrain=config.network.init,
                                       joint=config.network.joint)

    if config.network.sim_header == "tightwc":
        fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)
    else:
        fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    local_model_path = "./mamba_model"
    model_text = MambaTextEncoder(local_model_path, output_dim=512)

    # 使用 ImageCLIP 包装器
    model_image = ImageCLIP(model)

    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    model_text = torch.nn.DataParallel(model_text).cuda()

    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, num_segments=config.data.num_segments,
                               image_tmpl=config.data.image_tmpl,
                               transform=get_augmentation(False, config), random_shift=False)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=False, drop_last=False)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain)))
            checkpoint = torch.load(config.pretrain, map_location=device, weights_only=True)

            # 对于 model (无 DataParallel 前缀)，使用 remove_prefix
            model.load_state_dict(remove_dataparallel_prefix(checkpoint['model_state_dict']), strict=False)

            if 'fusion_model_state_dict' in checkpoint:
                # 对于 fusion_model (有 DataParallel)，不 remove_prefix，直接加载
                fusion_model.load_state_dict(checkpoint['fusion_model_state_dict'], strict=False)
            else:
                print("Warning: 'fusion_model_state_dict' not found in checkpoint.")

            # 加载 Mamba 模型，对于 model_text (有 DataParallel)，不 remove_prefix，直接加载
            if 'model_text_state_dict' in checkpoint:
                model_text.load_state_dict(checkpoint['model_text_state_dict'], strict=False)
            else:
                print("Warning: 'model_text_state_dict' not found in checkpoint. Mamba model will not be loaded.")

            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.pretrain)))

    classes, num_text_aug, text_dict = mamba_text_prompt_with_descriptions(val_data)

    if config.solver.evaluate:
        # 传递 model_image (DataParallel 包装的 ImageCLIP) 给 validate 函数
        validate(start_epoch, val_loader, classes, device, model_image, fusion_model, config, model_text)
    else:
        # 如果不是只评估模式，也提供一个调用示例
        validate(start_epoch, val_loader, classes, device, model_image, fusion_model, config, model_text)


if __name__ == '__main__':
    main()
