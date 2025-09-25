import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import h5py
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import cv2
import json

def main():
    
    default_prompt = "Pick up the PCB board on the round yellow base and place it into the circular recess of the yellow square container."
    repo_id = "lerobot/test"
    root = "/home/agx/jedata/test_0807a_modified"
    export_root="/home/agx/jedata/hdf5_0807a_modified/"
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
            prompt = step["task"]

            # 四个相机分别编码为jpg
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
                success, img_encoded = cv2.imencode('.jpg', img)
                if not success:
                    print(f"[imencode failed] eid={eid}, t={t}, cam={cam}, shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
                    img_list.append(b'')
                else:
                    img_list.append(img_encoded.tobytes())

        states = np.array(states)
        actions = np.array(actions)

        # 保存为 hdf5
        hdf5_path = export_root + f'lerobot_episode_{eid}.hdf5'
        with h5py.File(hdf5_path, 'w') as f:
            obs_grp = f.create_group('observations')
            obs_grp.create_dataset('qpos', data=states)
            img_grp = obs_grp.create_group('images')
            dt = h5py.special_dtype(vlen=np.dtype('uint8'))
            img_grp.create_dataset('top_rgb', (len(top_rgb),), dtype=dt)
            img_grp.create_dataset('right_wrist_rgb', (len(right_wrist_rgb),), dtype=dt)
            img_grp.create_dataset('right_pole_rgb', (len(right_pole_rgb),), dtype=dt)
            img_grp.create_dataset('base_rgb', (len(base_rgb),), dtype=dt)
            for i in range(len(top_rgb)):
                img_grp['top_rgb'][i] = np.frombuffer(top_rgb[i], dtype='uint8')
                img_grp['right_wrist_rgb'][i] = np.frombuffer(right_wrist_rgb[i], dtype='uint8')
                img_grp['right_pole_rgb'][i] = np.frombuffer(right_pole_rgb[i], dtype='uint8')
                img_grp['base_rgb'][i] = np.frombuffer(base_rgb[i], dtype='uint8')
            f.create_dataset('action', data=actions)
        print(f"Saved episode {eid} to {hdf5_path}")

        # 保存 instruction json
        instr_dict = {
            "instruction": prompt,
            "simplified_instruction": prompt,
            "expanded_instruction": prompt
        }
        json_path = os.path.join(os.path.dirname(hdf5_path), 'expanded_instruction_gpt-4-turbo.json')
        with open(json_path, 'w') as jf:
            json.dump(instr_dict, jf)

if __name__ == "__main__":
    main()