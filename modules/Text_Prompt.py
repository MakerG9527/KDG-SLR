import torch
import clip
import pandas as pd
import os
from modules.mamba import MambaTextEncoder


def text_prompt_with_descriptions(data):
    """
    此函数现在从 Action_DATASETS 对象中获取标签文件路径，并处理列不存在的情况。
    """
    labels_file = data.labels_file

    # 检查文件路径是否存在
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Label file not found at: {labels_file}")

    # 使用 header=None 读取，以防文件没有头部
    all_labels_df = pd.read_csv(labels_file, header=None)

    class_to_description = {}

    # 检查列数
    num_cols = all_labels_df.shape[1]

    for _, row in all_labels_df.iterrows():
        # 确保索引1是类别名
        class_name = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else ""

        # 修正: 检查列数，如果不足4列，则使用类别名作为描述
        if num_cols >= 4:
            description = row.iloc[3] if not pd.isna(row.iloc[3]) else class_name
        else:
            description = class_name

        # 确保description是字符串类型
        if not isinstance(description, str):
            description = str(description)

        # 处理空描述的情况
        if not description.strip():
            description = class_name

        class_to_description[class_name] = description

    # 获取数据集中的类别列表
    dataset_classes = [str(c[0]) for c in data.classes]

    # 只使用一个固定的文本格式，不使用数据增强模板
    text_dict = {}
    num_text_aug = 1  # 只有一种文本表示

    # 直接使用描述作为文本
    descriptions = []
    for class_name in dataset_classes:
        descriptions.append(class_to_description.get(class_name, class_name))

    # 确保所有描述都是字符串，并处理可能的异常值
    tokenized_texts = []
    for i, desc in enumerate(descriptions):
        try:
            if not isinstance(desc, str):
                desc = str(desc)
            if desc is not None and desc.strip():
                tokenized_texts.append(clip.tokenize(desc))
            else:
                tokenized_texts.append(clip.tokenize(dataset_classes[i]))
        except Exception as e:
            print(f"Warning: Error tokenizing description at index {i}: '{desc}' - {e}")
            # 使用类别名作为回退选项
            fallback_desc = dataset_classes[i]
            tokenized_texts.append(clip.tokenize(fallback_desc))

    text_dict[0] = torch.cat(tokenized_texts)

    classes = text_dict[0]
    return classes, num_text_aug, text_dict


def mamba_text_prompt_with_descriptions(data):
    """
    此函数现在从 Action_DATASETS 对象中获取标签文件路径，并处理列不存在的情况。
    """
    labels_file = '/home/newdisk2/gld/ActionCLIP/lists/all_labels.csv'

    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Label file not found at: {labels_file}")

    all_labels_df = pd.read_csv(labels_file, header=None)

    class_to_description = {}
    for _, row in all_labels_df.iterrows():
        class_name = str(row.iloc[1]) if not pd.isna(row.iloc[1]) else ""
        description = row.iloc[3] if not pd.isna(row.iloc[3]) else class_name
        if not isinstance(description, str):
            description = str(description)
        if not description.strip():
            description = class_name
        class_to_description[class_name] = description

    dataset_classes = [str(c[0]) for c in data.classes]
    descriptions = []
    for class_name in dataset_classes:
        descriptions.append(class_to_description.get(class_name, class_name))

    num_text_aug = 1
    text_dict = descriptions

    return descriptions, num_text_aug, descriptions