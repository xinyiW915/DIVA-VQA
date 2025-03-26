import os
import cv2
import pandas as pd
import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils import data


class VideoDataset_feature(data.Dataset):
    def __init__(self, data_dir, filename_path, transform, resize, database, patch_size=16, target_size=224, top_n=196):
        super(VideoDataset_feature, self).__init__()
        if isinstance(filename_path, str):
            dataInfo = pd.read_csv(filename_path)
        elif isinstance(filename_path, pd.DataFrame):
            dataInfo = filename_path
        else:
            raise ValueError("filename_path: CSV file or DataFrame")
        self.video_names = dataInfo['vid'].tolist()
        self.transform = transform
        self.videos_dir = data_dir
        self.resize = resize
        self.database = database
        self.patch_size = patch_size
        self.target_size = target_size
        self.top_n = top_n
        self.length = len(self.video_names)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.database == 'konvid_1k' or self.database == 'test':
            video_clip_min = 8
            video_name = str(self.video_names[idx]) + '.mp4'
        elif self.database == 'live_vqc':
            video_clip_min = 10
            video_name = str(self.video_names[idx]) + '.mp4'
        elif self.database == 'cvd_2014':
            video_clip_min = 12
            video_name = str(self.video_names[idx]) + '.avi'
        elif self.database == 'youtube_ugc':
            video_clip_min = 20
            video_name = str(self.video_names[idx]) + '.mkv'
        elif self.database == 'youtube_ugc_h264':
            video_clip_min = 20
            video_name = str(self.video_names[idx]) + '.mp4'
        elif self.database == 'lsvq_test_1080p' or self.database == 'lsvq_test' or self.database == 'lsvq_train':
            video_clip_min = 8
            video_name = str(self.video_names[idx]) + '.mp4'

        filename = os.path.join(self.videos_dir, video_name)

        video_capture = cv2.VideoCapture(filename)
        video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        if not video_capture.isOpened():
            raise RuntimeError(f"Failed to open video: {filename}")

        video_channel = 3
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frame_rate = int(round(video_capture.get(cv2.CAP_PROP_FPS)))
        video_clip = int(video_length / video_frame_rate) if video_frame_rate != 0 else 10
        video_length_clip = 32
        # print(video_length)
        # print(video_frame_rate)
        # print(video_clip)

        all_frame_tensor = torch.zeros((video_length, video_channel, self.resize, self.resize), dtype=torch.float32)
        all_residual_frag_tensor = torch.zeros((video_length - 1, video_channel, self.resize, self.resize), dtype=torch.float32)
        all_frame_frag_tensor = torch.zeros((video_length - 1, video_channel, self.resize, self.resize), dtype=torch.float32)

        video_read_index = 0
        prev_frame = None
        for i in range(video_length):
            has_frames, frame = video_capture.read()
            if has_frames:
                curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # save_img(curr_frame, fig_path='original', img_title=f'original_{i}')

                # frame features
                curr_frame_tensor = self.transform(Image.fromarray(curr_frame))
                all_frame_tensor[video_read_index] = curr_frame_tensor

                # frame frag features
                if prev_frame is not None:
                    residual = cv2.absdiff(curr_frame, prev_frame)
                    # save_img(residual, fig_path='residual', img_title=f'residual_{i}')

                    diff = self.get_patch_diff(residual)
                    # frame residual fragment
                    imp_patches, positions = self.extract_important_patches(residual, diff)
                    imp_patches_pil = Image.fromarray(imp_patches.astype('uint8'))
                    # save_img(imp_patches_pil, fig_path='residual_frag', img_title=f'residual_frag_{i}')

                    residual_frag_tensor = self.transform(imp_patches_pil)
                    all_residual_frag_tensor[video_read_index] = residual_frag_tensor

                    # current frame fragment
                    ori_patches = self.get_original_frame_patches(curr_frame, positions)
                    ori_patches_pil = Image.fromarray(ori_patches.astype('uint8'))
                    # save_img(ori_patches_pil, fig_path='ori_frag', img_title=f'ori_frag_{i}')

                    frame_frag_tensor = self.transform(ori_patches_pil)
                    all_frame_frag_tensor[video_read_index] = frame_frag_tensor

                    video_read_index += 1
            prev_frame = curr_frame
        video_capture.release()
        # visualisation
        visualise_image(curr_frame, 'Current Frame')
        visualise_image(imp_patches_pil, 'Residual Fragment')
        visualise_image(ori_patches_pil, 'Frame Fragment')

        # Unfilled frames
        self.fill_tensor(all_frame_tensor, video_read_index, video_length)
        self.fill_tensor(all_residual_frag_tensor, video_read_index, video_length - 1)
        self.fill_tensor(all_frame_frag_tensor, video_read_index, video_length - 1)

        video_all = []
        video_res_frag_all = []
        video_frag_all = []
        for i in range(video_clip):
            clip_tensor = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            clip_res_frag_tensor = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])
            clip_frag_tensor = torch.zeros([video_length_clip, video_channel, self.resize, self.resize])

            start_idx = i * video_frame_rate
            end_idx = start_idx + video_length_clip
            # frame features
            if end_idx <= video_length:
                clip_tensor = all_frame_tensor[start_idx:end_idx]
            else:
                clip_tensor[:(video_length - start_idx)] = all_frame_tensor[start_idx:]
                clip_tensor[(video_length - start_idx):video_length_clip] = clip_tensor[video_length - start_idx - 1]

            # frame frag features
            if end_idx <= (video_length - 1):
                clip_res_frag_tensor = all_residual_frag_tensor[start_idx:end_idx]
                clip_frag_tensor = all_frame_frag_tensor[start_idx:end_idx]
            else:
                clip_res_frag_tensor[:(video_length - 1 - start_idx)] = all_residual_frag_tensor[start_idx:]
                clip_frag_tensor[:(video_length - 1 - start_idx)] = all_frame_frag_tensor[start_idx:]
                clip_res_frag_tensor[(video_length - 1 - start_idx):video_length_clip] = clip_res_frag_tensor[video_length - 1 - start_idx - 1]
                clip_frag_tensor[(video_length - 1 - start_idx):video_length_clip] = clip_frag_tensor[video_length - 1 - start_idx - 1]

            video_all.append(clip_tensor)
            video_res_frag_all.append(clip_res_frag_tensor)
            video_frag_all.append(clip_frag_tensor)

        # Underfilling of clips
        if video_clip < video_clip_min:
            for i in range(video_clip, video_clip_min):
                video_all.append(video_all[video_clip - 1])
                video_res_frag_all.append(video_res_frag_all[video_clip - 1])
                video_frag_all.append(video_frag_all[video_clip - 1])
        return video_all, video_res_frag_all, video_frag_all, video_name

    @staticmethod
    # duplicat the final frames
    def fill_tensor(tensor, read_index, length):
        if read_index < length:
            tensor[read_index:length] = tensor[read_index - 1]

    def get_patch_diff(self, residual_frame):
        h, w = residual_frame.shape[:2]
        patch_size = self.patch_size
        h_adj = (h // patch_size) * patch_size
        w_adj = (w // patch_size) * patch_size
        residual_frame_adj = residual_frame[:h_adj, :w_adj]
        # calculate absolute patch difference
        diff = np.zeros((h_adj // patch_size, w_adj // patch_size))
        for i in range(0, h_adj, patch_size):
            for j in range(0, w_adj, patch_size):
                patch = residual_frame_adj[i:i+patch_size, j:j+patch_size]
                # absolute sum
                diff[i // patch_size, j // patch_size] = np.sum(np.abs(patch))
        return diff

    def extract_important_patches(self, residual_frame, diff):
        patch_size = self.patch_size
        target_size = self.target_size
        top_n = self.top_n

        # find top n patches indices
        patch_idx = np.unravel_index(np.argsort(-diff.ravel()), diff.shape)
        top_patches = list(zip(patch_idx[0][:top_n], patch_idx[1][:top_n]))
        sorted_idx = sorted(top_patches, key=lambda x: (x[0], x[1]))

        imp_patches_img = np.zeros((target_size, target_size, residual_frame.shape[2]), dtype=residual_frame.dtype)
        patches_per_row = target_size // patch_size  # 14
        # order the patch in the original location relation
        positions = []
        for idx, (y, x) in enumerate(sorted_idx):
            patch = residual_frame[y * patch_size:(y + 1) * patch_size, x * patch_size:(x + 1) * patch_size]
            # new patch location
            row_idx = idx // patches_per_row
            col_idx = idx % patches_per_row
            start_y = row_idx * patch_size
            start_x = col_idx * patch_size
            imp_patches_img[start_y:start_y + patch_size, start_x:start_x + patch_size] = patch
            positions.append((y, x))
        return imp_patches_img, positions

    def get_original_frame_patches(self, original_frame, positions):
        patch_size = self.patch_size
        target_size = self.target_size
        imp_original_patches_img = np.zeros((target_size, target_size, original_frame.shape[2]), dtype=original_frame.dtype)
        patches_per_row = target_size // patch_size

        for idx, (y, x) in enumerate(positions):
            start_y = y * patch_size
            start_x = x * patch_size
            end_y = start_y + patch_size
            end_x = start_x + patch_size

            patch = original_frame[start_y:end_y, start_x:end_x]
            row_idx = idx // patches_per_row
            col_idx = idx % patches_per_row
            target_start_y = row_idx * patch_size
            target_start_x = col_idx * patch_size

            imp_original_patches_img[target_start_y:target_start_y + patch_size,
                                     target_start_x:target_start_x + patch_size] = patch
        return imp_original_patches_img

def visualise_tensor(tensors, num_frames_to_visualise=5, img_title='Frag'):
    np_feat = tensors.numpy()
    fig, axes = plt.subplots(1, num_frames_to_visualise, figsize=(15, 5))
    for i in range(num_frames_to_visualise):
        # move channels to last dimension for visualisation: (height, width, channels)
        frame = np_feat[i].transpose(1, 2, 0)
        # normalize to [0, 1] for visualisation
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        axes[i].imshow(frame)
        axes[i].axis('off')
        axes[i].set_title(f'{img_title} {i + 1}')

    plt.tight_layout()
    save_path =  f'../../figs/{img_title}.png'
    plt.savefig(save_path, dpi=300)
    plt.show()

def visualise_image(frame, img_title='Residual Fragment', debug=False):
    if debug:
        plt.figure(figsize=(5, 5))
        plt.imshow(frame)
        plt.axis('off')
        plt.title(img_title)
        plt.show()

def save_img(frame, fig_path, img_title):
    from torchvision.transforms import ToPILImage
    save_path = f'../../figs/{fig_path}/{img_title}.png'
    if isinstance(frame, torch.Tensor):
        if frame.dim() == 3 and frame.size(0) in [1, 3]:
            frame = ToPILImage()(frame)
        else:
            raise ValueError("Unsupported tensor shape. Expected shape (C, H, W) with C=1 or C=3.")

    if save_path:
        if isinstance(frame, torch.Tensor):
            frame = ToPILImage()(frame)
        elif isinstance(frame, np.ndarray):
            frame = Image.fromarray(frame)
        frame.save(save_path)
        print(f"Image saved to {save_path}")


if __name__ == "__main__":
    database = 'konvid_1k'
    videos_dir = '../../ugc_original_videos/'
    metadata_csv = '../../metadata/TEST_metadata.csv'
    # videos_dir = '/home/xinyi/video_dataset/KoNViD_1k/KoNViD_1k_videos/'
    # videos_dir = '/media/on23019/server/LSVQ/'
    # metadata_csv = f'../../metadata/{database.upper()}_metadata.csv'

    resize = 224 # 224, 384
    start_time = time.time()
    resize_transform = transforms.Compose([transforms.Resize([resize, resize]),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])])

    dataset = VideoDataset_feature(
        data_dir=videos_dir,
        filename_path=metadata_csv,
        transform=resize_transform,
        resize=resize,
        database=database,
        patch_size=16, # 8, 16, 32, 16, 32
        target_size=224, # 224, 224, 224, 384, 384
        top_n=14*14 # 28*28, 14*14, 7*7, 24*24, 12*12
    )

    # test
    index = 0
    video_segments, video_res_frag_all, video_frag_all, video_name = dataset[index]
    print(f"Video Name: {video_name}")
    print(f"Number of Video Segments: {len(video_segments)}")
    print(f"Number of Video Residual Fragment Segments: {len(video_res_frag_all)}")
    print(f"Number of Video Fragment Segments: {len(video_frag_all)}")
    print(f"Shape of Each Segment: {video_segments[0].shape}")  # (video_length_clip, channels, height, width)
    print(f"Shape of Each Residual Fragment Segments: {video_res_frag_all[0].shape}")
    print(f"Shape of Each Fragment Segments: {video_frag_all[0].shape}")

    # visualisation
    first_segments = video_segments[0]
    visualise_tensor(first_segments, num_frames_to_visualise=5, img_title='Frame')

    first_segment_residuals = video_res_frag_all[0]
    visualise_tensor(first_segment_residuals, num_frames_to_visualise=6, img_title='Residual Frag')

    first_segment_fragments = video_frag_all[0]
    visualise_tensor(first_segment_fragments, num_frames_to_visualise=5, img_title='Frame Frag')
    print(f"Processed {time.time() - start_time:.2f} seconds")