import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import models, transforms

from extractor.extract_rf_feats import VideoDataset_feature
from extractor.extract_slowfast_clip import SlowFast, extract_features_slowfast_pool
from extractor.extract_swint_clip import SwinT, extract_features_swint_pool
from model_regression import Mlp, preprocess_data


def get_transform(resize):
    return transforms.Compose([transforms.Resize([resize, resize]),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])

def setup_device(config):
    if config.device == "gpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.set_device(0)
    else:
        device = torch.device("cpu")
    print(f"Running on {'GPU' if device.type == 'cuda' else 'CPU'}")
    return device

def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            name = k[7:]
        elif k == 'n_averaged':
            continue
        else:
            name = k
        new_state_dict[name] = v
    return new_state_dict

def load_model(config, device, input_features=11520):
    model = Mlp(input_features=input_features, out_features=1, drop_rate=0.1, act_layer=nn.GELU).to(device)
    if config.is_finetune:
        model_path = os.path.join(config.save_path, f"reported_results/best_model/{config.test_data_name}_{config.network_name}_fine_tuned_model.pth")
    else:
        model_path = os.path.join(config.save_path, f"best_model/{config.train_data_name}_{config.network_name}_{config.model_name}_{config.select_criteria}"
                                                       f"_trained_median_model_param_kfold.pth")
    # print("Loading model from:", model_path)
    state_dict = torch.load(model_path, map_location=device)
    fixed_state_dict = fix_state_dict(state_dict)
    try:
        model.load_state_dict(fixed_state_dict)
    except RuntimeError as e:
        print(e)
    return model

def evaluate_video_quality(config, data_loader, model_slowfast, model_swint, model_mlp):
    # get video features
    model_slowfast.eval()
    model_swint.eval()
    with torch.no_grad():
        for i, (video_segments, video_res_frag_all, video_frag_all, video_name) in enumerate(tqdm(data_loader, desc="Processing Videos")):
            # slowfast features
            _, _, slowfast_frame_feats = extract_features_slowfast_pool(video_segments, model_slowfast, device)
            _, _, slowfast_res_frag_feats = extract_features_slowfast_pool(video_res_frag_all, model_slowfast, device)
            _, _, slowfast_frame_frag_feats = extract_features_slowfast_pool(video_frag_all, model_slowfast, device)
            slowfast_frame_feats_avg = slowfast_frame_feats.mean(dim=0)
            slowfast_res_frag_feats_avg = slowfast_res_frag_feats.mean(dim=0)
            slowfast_frame_frag_feats_avg = slowfast_frame_frag_feats.mean(dim=0)

            # swinT feature
            swint_frame_feats = extract_features_swint_pool(video_segments, model_swint, device)
            swint_res_frag_feats = extract_features_swint_pool(video_res_frag_all, model_swint, device)
            swint_frame_frag_feats = extract_features_swint_pool(video_frag_all, model_swint, device)
            swint_frame_feats_avg = swint_frame_feats.mean(dim=0)
            swint_res_frag_feats_avg = swint_res_frag_feats.mean(dim=0)
            swint_frame_frag_feats_avg = swint_frame_frag_feats.mean(dim=0)

            # frame + residual fragment + frame fragment features
            rf_vqa_feats = torch.cat((slowfast_frame_feats_avg, slowfast_res_frag_feats_avg, slowfast_frame_frag_feats_avg,
                                      swint_frame_feats_avg, swint_res_frag_feats_avg, swint_frame_frag_feats_avg), dim=0)

    rf_vqa_feats = rf_vqa_feats
    feature_tensor, _ = preprocess_data(rf_vqa_feats, None)
    if feature_tensor.dim() == 1:
        feature_tensor = feature_tensor.unsqueeze(0)
    # print(f"Feature tensor shape before MLP: {feature_tensor.shape}")

    model_mlp.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prediction = model_mlp(feature_tensor)
            predicted_score = prediction.item()
            return predicted_score

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', type=str, default='gpu', help='cpu or gpu')
    parser.add_argument('-model_name', type=str, default='Mlp', help='Name of the regression model')
    parser.add_argument('-select_criteria', type=str, default='byrmse', help='Selection criteria')
    parser.add_argument('-is_finetune', type=bool, default=True, help='With or without finetune')
    parser.add_argument('-save_path', type=str, default='../log/', help='Path to save models')

    parser.add_argument('-train_data_name', type=str, default='lsvq_train', help='Name of the training data')
    parser.add_argument('-test_data_name', type=str, default='konvid_1k', help='Name of the testing data')
    parser.add_argument('-test_video_path', type=str, default='../ugc_original_videos/video_test_time/5636101558_540p.mp4', help='demo test video')

    parser.add_argument('--network_name', type=str, default='diva-vqa_large')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=224, help='224, 384')
    parser.add_argument('--patch_size', type=int, default=16, help='8, 16, 32, 8, 16, 32')
    parser.add_argument('--target_size', type=int, default=224, help='224, 224, 224, 384, 384, 384')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = parse_arguments()
    device = setup_device(config)

    # test demo video
    resize_transform = get_transform(config.resize)
    top_n = int(config.target_size /config. patch_size) * int(config.target_size / config.patch_size)
    data = {'vid': [os.path.splitext(os.path.basename(config.test_video_path))[0]],
        'test_data_name': [config.test_data_name],
        'test_video_path': [config.test_video_path]}
    videos_dir = os.path.dirname(config.test_video_path)
    test_df = pd.DataFrame(data)
    # print(test_df.T)

    dataset = VideoDataset_feature(videos_dir, test_df, resize_transform, config.resize, config.test_data_name, config.patch_size, config.target_size, top_n)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=min(config.num_workers, os.cpu_count()), pin_memory=True
    )
    # print(f"Dataset loaded. Total videos: {len(dataset)}, Total batches: {len(data_loader)}")

    # load models to device
    model_slowfast = SlowFast().to(device)
    if config.network_name == 'diva-vqa':
        model_swint = SwinT(global_pool='avg').to(device) # 'swin_base_patch4_window7_224.ms_in22k_ft_in1k'
        input_features = 9984
    elif config.network_name == 'diva-vqa_large':
        model_swint = SwinT(model_name='swin_large_patch4_window7_224', global_pool='avg', pretrained=True).to(device)
        input_features = 11520
    model_mlp = load_model(config, device, input_features)

    quality_prediction = evaluate_video_quality(config, data_loader, model_slowfast, model_swint, model_mlp)
    print("Predicted Quality Score:", quality_prediction)