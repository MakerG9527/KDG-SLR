# 在 saving.py 中
import torch

def epoch_saving(epoch, model, fusion_model, optimizer, model_text, filename):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'fusion_model_state_dict': fusion_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # 修复：正确传递 model_text 参数
                    'model_text_state_dict': model_text.state_dict(),
                    }, filename) #just change to your preferred folder/filename

def best_saving(working_dir, epoch, model, fusion_model, optimizer, model_text):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'fusion_model_state_dict': fusion_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 修复：正确传递 model_text 参数
        'model_text_state_dict': model_text.state_dict(),
    }, best_name)  # just change to your preferred folder/filename
