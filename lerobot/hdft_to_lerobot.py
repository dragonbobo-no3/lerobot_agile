import h5py
import numpy as np
import time
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def convert_hdf5_to_lerobot(
    hdf5_path,
    lerobot_root,
    repo_id="dragonbobo-no3/lerobot_dataset",
    fps=30,
    use_videos=False,
    task="test_task",
):
    t0 = time.time()
    print(f"[INFO] 开始读取 hdf5: {hdf5_path}")
    with h5py.File(hdf5_path, 'r') as f:
        actions = f['action'][...]              # (batch, 7)
        qpos = f['observations/qpos'][...]      # (batch, 7)
        images_grp = f['observations/images']
        batch_size = actions.shape[0]
        camera_names = list(images_grp.keys())
        images_dict = {cam: images_grp[cam][...] for cam in camera_names}
    t1 = time.time()
    print(f"[INFO] hdf5读取完成，batch_size: {batch_size}，摄像头: {camera_names}，耗时: {t1-t0:.3f}s")

    # 构造 features，自定义名称
    joint_prefix = "right.joint"
    joint_suffix = ".pos"
    joint_num = actions.shape[1]

    action_names = [f"{joint_prefix}{i}{joint_suffix}" for i in range(joint_num)]
    state_names = [f"{joint_prefix}{i}{joint_suffix}" for i in range(joint_num)]

    features = {
        "action": {"dtype": "float32", "shape": [joint_num], "names": action_names},
        "observation.state": {"dtype": "float32", "shape": [joint_num], "names": state_names},
    }
    for cam in camera_names:
        img_shape = images_dict[camera_names[0]][0].shape  # (H, W, C)
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": list(img_shape),
            "names": ["height", "width", "channels"]
        }
    print(f"[INFO] 自动构造 features: {features}")

    # 初始化 LerobotDataset
    t2 = time.time()
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=lerobot_root,
        use_videos=use_videos,
        image_writer_processes=1,
        image_writer_threads=1 * len(camera_names),
    )
    t3 = time.time()
    print(f"[INFO] LerobotDataset 初始化完成，耗时: {t3-t2:.3f}s")

    # 构造 episode_data
    episode_data = {
        "size": batch_size,
        "action": [actions[i] for i in range(batch_size)],
        "observation.state": [qpos[i] for i in range(batch_size)],
        "frame_index": list(range(batch_size)),
        "timestamp": [float(i) / fps for i in range(batch_size)],
        "task": ["test_task"] * batch_size,
        "episode_index": 0,  # 或自动推断
        "task_index": [0] * batch_size,  # 新增，兼容 LeRobotDataset
        "index": list(range(batch_size)), # 新增，兼容 LeRobotDataset
    }
    for cam in camera_names:
        if use_videos:
            # 视频模式，保持 (N, C, H, W) 或 (C, H, W)
            episode_data[f"observation.images.{cam}"] = [
                np.transpose(images_dict[cam][i], (2, 0, 1)) for i in range(batch_size)
            ]
        else:
            # 图片模式，直接用 (H, W, C)
            episode_data[f"observation.images.{cam}"] = [
                images_dict[cam][i] for i in range(batch_size)
        ]

    print("[INFO] episode_data 字段:")
    for k, v in episode_data.items():
        if isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'shape'):
            print(f"  {k}: list, shape[0]={v[0].shape}, len={len(v)}")
        else:
            print(f"  {k}: {type(v)}, len={len(v) if hasattr(v,'__len__') else 'N/A'}")

    # 存储 episode
    t4 = time.time()
    dataset.save_episode(episode_data=episode_data)
    t5 = time.time()
    print(f"[INFO] Saved lerobot episode to {lerobot_root}，耗时: {t5-t4:.3f}s")

if __name__ == "__main__":
    hdf5_path = "/home/agx/jedata/hdf5_expanded/lerobot_episode_10_expanded.hdf5"
    lerobot_root = "/home/agx/jedata/wrapper"
    convert_hdf5_to_lerobot(hdf5_path, lerobot_root)