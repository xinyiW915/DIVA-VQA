import argparse
import os
import time
import logging
import torch
from torchvision import transforms
from tqdm import tqdm

from extractor.extract_slowfast_clip import SlowFast, extract_features_slowfast_pool
from extractor.extract_swint_clip import SwinT, extract_features_swint_pool
from utils.logger_setup import setup_logging

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    logging.info(f"Using device: {device}")
    return device

def get_transform(resize):
    return transforms.Compose([transforms.Resize([resize, resize]),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])

def load_dataset(resize_transform, resize, database, patch_size, target_size, top_n):
    if database == 'konvid_1k':
        videos_dir = '/media/on23019/server1/video_dataset/KoNViD_1k/KoNViD_1k_videos/'
        # videos_dir = 'D:/video_dataset/KoNViD_1k/KoNViD_1k_videos/'
    elif database == 'live_vqc':
        videos_dir = '/media/on23019/server1/video_dataset/LIVE-VQC/Video/'
        # videos_dir = 'D:/video_dataset/LIVE-VQC/video/'
    elif database == 'cvd_2014':
        videos_dir = '/media/on23019/server1/video_dataset/CVD2014/'
        # videos_dir = 'D:/video_dataset/CVD2014/'
    elif database == 'youtube_ugc':
        videos_dir = '/media/on23019/server1/video_dataset/youtube_ugc/all/'
        # videos_dir = 'D:/video_dataset/ugc-dataset/youtube_ugc/original_videos/all/'
    elif database == 'youtube_ugc_h264':
        videos_dir = '/media/on23019/server1/video_dataset/youtube_ugc/all_h264/'
        # videos_dir = 'D:/video_dataset/ugc-dataset/youtube_ugc/original_videos/all_h264/'
    elif database == 'lsvq_test_1080p' or database =='lsvq_test' or database == 'lsvq_train':
        videos_dir = '/media/on23019/server1/LSVQ/'
    elif database == 'test':
        videos_dir = '../ugc_original_videos/'
    metadata_csv = f'../metadata/{database.upper()}_metadata.csv'

    # split test: temp
    # metadata_csv = f'../metadata/{database.upper()}_metadata_part3.csv'

    return VideoDataset_feature(videos_dir, metadata_csv, resize_transform, resize, database, patch_size, target_size, top_n)

def process_videos(data_loader, model_slowfast, model_swint, device):
    features_list = []
    model_slowfast.eval()
    model_swint.eval()

    with torch.no_grad():
        for i, (video_segments, video_res_frag_all, video_frag_all, video_name) in enumerate(tqdm(data_loader, desc="Processing Videos")):
            start_time = time.time()
            try:
                # slowfast features
                _, _, slowfast_frame_feats= extract_features_slowfast_pool(video_segments, model_slowfast, device)
                _, _, slowfast_res_frag_feats = extract_features_slowfast_pool(video_res_frag_all, model_slowfast, device)
                _, _, slowfast_frame_frag_feats = extract_features_slowfast_pool(video_frag_all, model_slowfast, device)
                slowfast_frame_feats_avg = slowfast_frame_feats.mean(dim=0)
                slowfast_res_frag_feats_avg = slowfast_res_frag_feats.mean(dim=0)
                slowfast_frame_frag_feats_avg = slowfast_frame_frag_feats.mean(dim=0)
                # logging.info(f"SlowFast Frame Feature shape: {slowfast_frame_feats_avg.shape}")
                # logging.info(f"SlowFast Residual Fragment Feature shape: {slowfast_res_frag_feats_avg.shape}")
                # logging.info(f"SlowFast Frame Fragment Feature shape: {slowfast_frame_frag_feats_avg.shape}")

                # swinT feature
                swint_frame_feats = extract_features_swint_pool(video_segments, model_swint, device)
                swint_res_frag_feats = extract_features_swint_pool(video_res_frag_all, model_swint, device)
                swint_frame_frag_feats = extract_features_swint_pool(video_frag_all, model_swint, device)
                swint_frame_feats_avg = swint_frame_feats.mean(dim=0)
                swint_res_frag_feats_avg = swint_res_frag_feats.mean(dim=0)
                swint_frame_frag_feats_avg = swint_frame_frag_feats.mean(dim=0)
                # logging.info(f"Swin-T Frame Feature shape: {swint_frame_feats_avg.shape}")
                # logging.info(f"Swin-T Residual Fragment Feature shape: {swint_res_frag_feats_avg.shape}")
                # logging.info(f"Swin-T Frame Fragment Feature shape: {swint_frame_frag_feats_avg.shape}")

                # frame + residual fragment + frame fragment features
                rf_vqa_feats = torch.cat((slowfast_frame_feats_avg, slowfast_res_frag_feats_avg, slowfast_frame_frag_feats_avg, swint_frame_feats_avg, swint_res_frag_feats_avg, swint_frame_frag_feats_avg), dim=0)
                logging.info(f"Feature shape: {rf_vqa_feats.shape}")
                features_list.append(rf_vqa_feats)

                logging.info(f"Processed {video_name[0]} in {time.time() - start_time:.2f} seconds")
                torch.cuda.empty_cache()

            except (KeyboardInterrupt, Exception) as e:
                if isinstance(e, KeyboardInterrupt):
                    logging.error("Processing interrupted by user.")
                else:
                    logging.error(f"Failed to process video {video_name[0]}: {e}")
            # finally:
            #     save_features(features_list, config.feature_save_path, config.feat_name, config.database)

    return features_list

def save_features(features_list, pt_path):
    if features_list:
        features_tensor = torch.stack(features_list)
        try:
            torch.save(features_tensor, f"{pt_path}")

        except Exception as e:
            print(f"Failed to save features: {e}")
        logging.info(f"Features saved to {pt_path}: {features_tensor.shape}\n")
    else:
        logging.warning("No features were processed. Nothing to save.")


def main(config):
    feature_save_path = os.path.abspath(os.path.join(config.feature_save_path, config.feat_name))
    pt_path = f'{feature_save_path}/{config.feat_name}_patch{config.patch_size}_{config.target_size}_top{config.top_n}_{config.database}_features.pt'
    print(pt_path)
    # split test: temp
    # pt_path = f'{feature_save_path}/{config.feat_name}_patch{config.patch_size}_{config.target_size}_top{config.top_n}_{config.database}_features_part3.pt'
    # print(pt_path)

    if not os.path.exists(feature_save_path):
        os.makedirs(feature_save_path)

    setup_logging(config.log_file)
    device = setup_device()
    resize_transform = get_transform(config.resize)
    dataset = load_dataset(resize_transform, config.resize, config.database, config.patch_size, config.target_size, config.top_n)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=min(config.num_workers, os.cpu_count()), pin_memory=True
    )
    logging.info(f"Dataset loaded. Total videos: {len(dataset)}, Total batches: {len(data_loader)}")

    model_slowfast = SlowFast().to(device)
    if config.feat_name == 'diva-vqa':
        model_swint = SwinT(global_pool='avg').to(device) # 'swin_base_patch4_window7_224.ms_in22k_ft_in1k'
    elif config.feat_name == 'diva-vqa_sub':
        model_swint = SwinT(global_pool='avg').to(device) # 'swin_base_patch4_window7_224.ms_in22k_ft_in1k'
    elif config.feat_name == 'diva-vqa_large':
        model_swint = SwinT(model_name='swin_large_patch4_window7_224', global_pool='avg', pretrained=True).to(device) # swin_large_patch4_window7_224.ms_in22k_ft_in1k
    elif config.feat_name == 'diva-vqa_small':
        model_swint = SwinT(model_name='swin_small_patch4_window7_224', global_pool='avg', pretrained=True).to(device) # swin_small_patch4_window7_224.ms_in22k_ft_in1k
    elif config.feat_name == 'diva-vqa_tiny':
        model_swint = SwinT(model_name='swin_tiny_patch4_window7_224', global_pool='avg', pretrained=True).to(device) # swin_tiny_patch4_window7_224.ms_in1k
    elif config.feat_name == 'diva-vqa_base_384':
        model_swint = SwinT(model_name='swin_base_patch4_window12_384', global_pool='avg', pretrained=True).to(device)  # swin_base_patch4_window12_384.ms_in22k_ft_in1k
    elif config.feat_name == 'diva-vqa_large_384':
        model_swint = SwinT(model_name='swin_large_patch4_window12_384', global_pool='avg', pretrained=True).to(device)  # swin_large_patch4_window12_384.ms_in22k_ft_in1k

    features_list = process_videos(data_loader, model_slowfast, model_swint, device)
    save_features(features_list, pt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='lsvq_train')
    parser.add_argument('--feat_name', type=str, default='diva-vqa_large')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resize', type=int, default=224, help='224, 384')
    parser.add_argument('--patch_size', type=int, default=16, help='8, 16, 32, 8, 16, 32')
    parser.add_argument('--target_size', type=int, default=224, help='224, 224, 224, 384, 384, 384')
    parser.add_argument('--top_n', type=int, default=14*14, help='28*28, 14*14, 7*7, 48*48, 24*24, 12*12')
    parser.add_argument('--feature_save_path', type=str, default=f"../features/diva-vqa/", help="../features/diva-vqa/, ../features_ablation/")
    parser.add_argument('--log_file', type=str, default="./utils/logging_diva-vqa_feats.log")


    config = parser.parse_args()
    if config.feat_name in ['diva-vqa', 'diva-vqa_large', 'diva-vqa_small', 'diva-vqa_tiny', 'diva-vqa_base_384', 'diva-vqa_large_384']:
        from extractor.extract_rf_feats import VideoDataset_feature
    elif config.feat_name in ['diva-vqa_sub']:
        from extractor.extract_rf_subsampling import VideoDataset_feature
    else:
        raise ValueError(f"Unknown feat_name: {config.feat_name}")

    print(config.feat_name)
    main(config)
