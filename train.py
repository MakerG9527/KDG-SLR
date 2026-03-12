import os
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
from modules.Visual_Prompt import visual_prompt
from modules.Text_Prompt import *
from test import validate
from utils.Augmentation import *
from utils.solver import _optimizer, _lr_scheduler
from utils.tools import *
from utils.saving import *
import clip
import torch
import torch.nn.functional as F
from modules.mamba import MambaTextEncoder


def remove_dataparallel_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class ImageCLIP(nn.Module):
    def __init__(self, model):
        super(ImageCLIP, self).__init__()
        self.model = model

    def forward(self, image):
        return self.model.encode_image(image)


def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'], config['data']['dataset'],
                               args.log_time)

    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('train.py', working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, clip_state_dict = clip.load(config.network.arch, device=device, jit=False, tsm=config.network.tsm,
                                       T=config.data.num_segments, dropout=config.network.drop_out,
                                       emb_dropout=config.network.emb_dropout, pretrain=config.network.init,
                                       joint=config.network.joint)

    if config.pretrain:
        if os.path.isfile(config.pretrain):
            print(f"=> loading checkpoint '{config.pretrain}'")
            # 使用 weights_only=True 提高安全性
            checkpoint = torch.load(config.pretrain, map_location='cpu', weights_only=True)

            model.load_state_dict(remove_dataparallel_prefix(checkpoint['model_state_dict']), strict=False)

            fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

            fusion_model.load_state_dict(remove_dataparallel_prefix(checkpoint['fusion_model_state_dict']),
                                         strict=False)

            del checkpoint
        else:
            print(f"=> no checkpoint found at '{config.pretrain}'")
            fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)
    else:
        fusion_model = visual_prompt(config.network.sim_header, clip_state_dict, config.data.num_segments)

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    print('train transforms: {}'.format(transform_train.transforms))
    print('val transforms: {}'.format(transform_val.transforms))

    local_model_path = "./mamba_model"
    model_text = MambaTextEncoder(local_model_path, output_dim=512)
    model_image = ImageCLIP(model)

    # 冻结视觉编码器参数
    print("Freezing visual encoder parameters...")
    for name, param in model.named_parameters():
        # 冻结所有视觉编码器参数
        if "visual" in name:
            param.requires_grad = False
    print("Visual encoder frozen.")

    model_text = torch.nn.DataParallel(model_text).cuda()
    model_image = torch.nn.DataParallel(model_image).cuda()
    fusion_model = torch.nn.DataParallel(fusion_model).cuda()

    # 打印可训练参数信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fusion_params = sum(p.numel() for p in fusion_model.parameters() if p.requires_grad)
    mamba_params = sum(p.numel() for p in model_text.parameters() if p.requires_grad)

    print(f"Total trainable parameters in ViT: {trainable_params}")
    print(f"Total trainable parameters in Fusion Model: {fusion_params}")
    print(f"Total trainable parameters in Mamba: {mamba_params}")

    train_data = Action_DATASETS(config.data.train_list, config.data.label_list, num_segments=config.data.num_segments,
                                 image_tmpl=config.data.image_tmpl, random_shift=config.data.random_shift,
                                 transform=transform_train)
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                              shuffle=True, pin_memory=False, drop_last=True)

    val_data = Action_DATASETS(config.data.val_list, config.data.label_list, random_shift=False,
                               num_segments=config.data.num_segments, image_tmpl=config.data.image_tmpl,
                               transform=transform_val)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers, shuffle=False,
                            pin_memory=False, drop_last=True)

    if device == "cpu":
        model_text.float()
        model_image.float()
    else:
        clip.model.convert_weights(model_image)

    start_epoch = config.solver.start_epoch

    classes, num_text_aug, text_dict = mamba_text_prompt_with_descriptions(train_data)

    optimizer = _optimizer(config, model, fusion_model, model_text)
    lr_scheduler = _lr_scheduler(config, optimizer)

    best_prec1 = 0.0
    if config.solver.evaluate:
        prec1 = validate(start_epoch, val_loader, classes, device, model_image, fusion_model, config, model_text)
        return

    for epoch in range(start_epoch, config.solver.epochs):
        model_image.train()
        model_text.train()
        fusion_model.train()
        for kkk, (images, list_id) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            images = images.view((-1, config.data.num_segments, 3) + images.size()[-2:])
            b, t, c, h, w = images.size()

            # 修复：处理 list_id 中的 tensor 元素
            texts = []
            for i in list_id:
                # 将 tensor 转换为 Python 整数
                idx = i.item() if torch.is_tensor(i) else int(i)
                texts.append(text_dict[idx])

            images = images.to(device).view(-1, c, h, w)

            image_embedding = model_image(images)
            image_embedding = image_embedding.view(b, t, -1)
            image_embedding = fusion_model(image_embedding)

            text_embedding = model_text(texts)
            text_embedding = text_embedding.half()

            if config.network.fix_text:
                text_embedding.detach_()

            logit_scale = model.logit_scale.exp()

            logits_per_image = logit_scale * image_embedding @ text_embedding.t()
            logits_per_text = logit_scale * text_embedding @ image_embedding.t()

            # 修复：使用实际的logits维度创建ground truth，解决DataParallel维度不匹配问题
            image_batch_size = logits_per_image.size(0)
            text_batch_size = logits_per_text.size(0)

            # 确保ground truth与logits维度匹配
            if image_batch_size != text_batch_size:
                # 当使用DataParallel时，取较小的batch size
                actual_batch_size = min(image_batch_size, text_batch_size)
                ground_truth = torch.arange(actual_batch_size, dtype=torch.long, device=device)

                # 调整logits维度以匹配
                if logits_per_image.size(0) > actual_batch_size:
                    logits_per_image = logits_per_image[:actual_batch_size, :actual_batch_size]
                if logits_per_text.size(0) > actual_batch_size:
                    logits_per_text = logits_per_text[:actual_batch_size, :actual_batch_size]
            else:
                ground_truth = torch.arange(image_batch_size, dtype=torch.long, device=device)

            total_loss = (F.cross_entropy(logits_per_image, ground_truth) +
                          F.cross_entropy(logits_per_text, ground_truth)) / 2

            # 使用新的打印格式，提供更详细的信息
            print("epoch", epoch)
            print({"train_total_loss": f"{total_loss.item():.5f}"})
            current_lr = optimizer.param_groups[0]['lr']
            print({"lr": f"{current_lr:.20f}"})

            total_loss.backward()

            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip.model.convert_weights(model)

        # 🌟 修复：将学习率更新移到epoch循环的末尾
        if config.solver.type != 'monitor':
            lr_scheduler.step()

        if epoch % config.logging.eval_freq == 0:
            prec1 = validate(epoch, val_loader, classes, device, model_image, fusion_model, config, model_text)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1, best_prec1))
        print('Saving:')
        filename = f"{working_dir}/last_model.pt"

        # 修复：传递 model_text 参数给保存函数
        epoch_saving(epoch, model, fusion_model, optimizer, model_text, filename)
        if is_best:
            best_saving(working_dir, epoch, model, fusion_model, optimizer, model_text)


if __name__ == '__main__':
    main()