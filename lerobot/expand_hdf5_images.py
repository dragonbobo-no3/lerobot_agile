import os
import h5py
import numpy as np
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

def decode_jpg_array(jpg_list, height=480, width=640):
    imgs = []
    for jpg_bytes in jpg_list:
        arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((height, width, 3), dtype=np.uint8)
        imgs.append(img)
    return np.stack(imgs, axis=0)


def expand_episode_images(hdf5_path, out_path, height=480, width=640):
    with h5py.File(hdf5_path, 'r') as f:
        imgs_grp = f['observations/images']
        expanded = {}
        for cam in ['top_rgb', 'right_wrist_rgb', 'right_pole_rgb', 'base_rgb']:
            jpg_list = [imgs_grp[cam][i].tobytes() for i in range(len(imgs_grp[cam]))]
            expanded[cam] = decode_jpg_array(jpg_list, height, width)
    with h5py.File(out_path, 'w') as f_out:
        for cam, arr in expanded.items():
            f_out.create_dataset(cam, data=arr, compression='gzip')
    print(f"Expanded images saved to {out_path}")


def main():
    repo_id = "lerobot/test"
    root = "/home/agx/jedata/test_0807a_modified"
    export_root = "/home/agx/jedata/hdf5_expanded/"
    os.makedirs(export_root, exist_ok=True)
    dataset = lerobot_dataset.LeRobotDataset(repo_id, root=root)

    episode_indices = [item["episode_index"].item() for item in dataset.hf_dataset]
    unique_episode_ids = sorted(set(episode_indices))

    for eid in unique_episode_ids:
        episode = [i for i, ep_idx in enumerate(episode_indices) if ep_idx == eid]
        states, actions = [], []
        top_rgb, right_wrist_rgb, right_pole_rgb, base_rgb = [], [], [], []

        for t, idx in enumerate(episode):
            step = dataset[idx]
            states.append(step["observation.state"])
            actions.append(step["action"])
            # 四个相机直接存原始 numpy 数组
            for cam, img_list in zip(
                ["observation.images.camera0", "observation.images.camera1", "observation.images.camera2", "observation.images.camera3"],
                [top_rgb, right_wrist_rgb, right_pole_rgb, base_rgb]
            ):
                img = step[cam]
                if hasattr(img, 'numpy'):
                    img = img.numpy()
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                if img.shape[0] == 3 and img.ndim == 3:
                    img = img.transpose(1, 2, 0)
                img_list.append(img)

        states = np.array(states)
        actions = np.array(actions)
        top_rgb = np.stack(top_rgb, axis=0)
        right_wrist_rgb = np.stack(right_wrist_rgb, axis=0)
        right_pole_rgb = np.stack(right_pole_rgb, axis=0)
        base_rgb = np.stack(base_rgb, axis=0)

        hdf5_path = export_root + f'lerobot_episode_{eid}_expanded.hdf5'
        with h5py.File(hdf5_path, 'w') as f:
            obs_grp = f.create_group('observations')
            obs_grp.create_dataset('qpos', data=states)
            img_grp = obs_grp.create_group('images')
            img_grp.create_dataset('top_rgb', data=top_rgb, compression='gzip')
            img_grp.create_dataset('right_wrist_rgb', data=right_wrist_rgb, compression='gzip')
            img_grp.create_dataset('right_pole_rgb', data=right_pole_rgb, compression='gzip')
            img_grp.create_dataset('base_rgb', data=base_rgb, compression='gzip')
            f.create_dataset('action', data=actions)
        print(f"Saved expanded episode {eid} to {hdf5_path}")

if __name__ == '__main__':
    main()
