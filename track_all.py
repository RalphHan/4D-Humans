import setproctitle

setproctitle.setproctitle('track')
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import os
import sys

sys.path.append('../YOLOv6')
import hydra
import torch
import numpy as np
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from phalp.configs.base import FullConfig
from phalp.models.hmar.hmr import HMR2018Predictor
from phalp.trackers.PHALP import PHALP
from phalp.utils import get_pylogger
from phalp.configs.base import CACHE_DIR
from phalp.utils.utils import FrameExtractor
import rotation_conversions as geometry
import multiprocessing
import time
from google.cloud import storage
from collections import defaultdict
import binascii
import dotenv

dotenv.load_dotenv()

TASK_SERVER = os.getenv("TASK_SERVER")
SPLASH_AUTH_TOKEN = os.getenv("SPLASH_AUTH_TOKEN")

assert TASK_SERVER

import requests

from hmr2.datasets.utils import expand_bbox_to_aspect_ratio
import json

warnings.filterwarnings('ignore')

log = get_pylogger(__name__)


class HMR2Predictor(HMR2018Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # Setup our new model
        from hmr2.models import download_models, load_hmr2

        # Download and load checkpoints
        download_models()
        model, _ = load_hmr2()

        self.model = model
        self.model.eval()

    def forward(self, x):
        hmar_out = self.hmar_old(x)
        batch = {
            'img': x[:, :3, :, :],
            'mask': (x[:, 3, :, :]).clip(0, 1),
        }
        model_out = self.model(batch)
        out = hmar_out | {
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out


class HMR2023TextureSampler(HMR2Predictor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        # Model's all set up. Now, load tex_bmap and tex_fmap
        # Texture map atlas
        bmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/bmap_256.npy'))
        fmap = np.load(os.path.join(CACHE_DIR, 'phalp/3D/fmap_256.npy'))
        self.register_buffer('tex_bmap', torch.tensor(bmap, dtype=torch.float))
        self.register_buffer('tex_fmap', torch.tensor(fmap, dtype=torch.long))

        self.img_size = 256  # self.cfg.MODEL.IMAGE_SIZE
        self.focal_length = 5000.  # self.cfg.EXTRA.FOCAL_LENGTH

        # import neural_renderer as nr
        # self.neural_renderer = nr.Renderer(dist_coeffs=None, orig_size=self.img_size,
        #                                   image_size=self.img_size,
        #                                   light_intensity_ambient=1,
        #                                   light_intensity_directional=0,
        #                                   anti_aliasing=False)

    def forward(self, x):
        x = x[:, :3, :, :]
        batch = {
            'img': x,
            # 'mask': (x[:,3,:,:]).clip(0,1),
        }
        model_out = self.model(batch)

        # from hmr2.models.prohmr_texture import unproject_uvmap_to_mesh

        def unproject_uvmap_to_mesh(bmap, fmap, verts, faces):
            # bmap:  256,256,3
            # fmap:  256,256
            # verts: B,V,3
            # faces: F,3
            valid_mask = (fmap >= 0)

            fmap_flat = fmap[valid_mask]  # N
            bmap_flat = bmap[valid_mask, :]  # N,3

            face_vids = faces[fmap_flat, :]  # N,3
            face_verts = verts[:, face_vids, :]  # B,N,3,3

            bs = face_verts.shape
            map_verts = torch.einsum('bnij,ni->bnj', face_verts, bmap_flat)  # B,N,3

            return map_verts, valid_mask

        pred_verts = model_out['pred_vertices'] + model_out['pred_cam_t'].unsqueeze(1)
        device = pred_verts.device
        face_tensor = torch.tensor(self.smpl.faces.astype(np.int64), dtype=torch.long, device=device)
        map_verts, valid_mask = unproject_uvmap_to_mesh(self.tex_bmap, self.tex_fmap, pred_verts, face_tensor)  # B,N,3

        # Project map_verts to image using K,R,t
        # map_verts_view = einsum('bij,bnj->bni', R, map_verts) + t # R=I t=0
        focal = self.focal_length / (self.img_size / 2)
        map_verts_proj = focal * map_verts[:, :, :2] / map_verts[:, :, 2:3]  # B,N,2
        # map_verts_depth = map_verts[:, :, 2] # B,N

        # Render Depth. Annoying but we need to create this
        # K = torch.eye(3, device=device)
        # K[0, 0] = K[1, 1] = self.focal_length
        # K[1, 2] = K[0, 2] = self.img_size / 2  # Because the neural renderer only support squared images
        # K = K.unsqueeze(0)
        # R = torch.eye(3, device=device).unsqueeze(0)
        # t = torch.zeros(3, device=device).unsqueeze(0)
        # rend_depth = self.neural_renderer(pred_verts,
        #                                 face_tensor[None].expand(pred_verts.shape[0], -1, -1).int(),
        #                                 # textures=texture_atlas_rgb,
        #                                 mode='depth',
        #                                 K=K, R=R, t=t)

        # rend_depth_at_proj = torch.nn.functional.grid_sample(rend_depth[:,None,:,:], map_verts_proj[:,None,:,:]) # B,1,1,N
        # rend_depth_at_proj = rend_depth_at_proj.squeeze(1).squeeze(1) # B,N
        img_rgba = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)  # B,4,H,W
        img_rgba_at_proj = torch.nn.functional.grid_sample(img_rgba, map_verts_proj[:, None, :, :])  # B,4,1,N
        img_rgba_at_proj = img_rgba_at_proj.squeeze(2)  # B,4,N

        # visibility_mask = map_verts_depth <= (rend_depth_at_proj + 1e-4) # B,N
        # img_rgba_at_proj[:,3,:][~visibility_mask] = 0

        # Paste image back onto square uv_image
        uv_image = torch.zeros((x.shape[0], 4, 256, 256), dtype=torch.float, device=device)
        uv_image[:, :, valid_mask] = img_rgba_at_proj

        out = {
            'uv_image': uv_image,
            'uv_vector': self.hmar_old.process_uv_image(uv_image),
            'pose_smpl': model_out['pred_smpl_params'],
            'pred_cam': model_out['pred_cam'],
        }
        return out


class HMR2_4dhuman(PHALP):
    def __init__(self, cfg):
        super().__init__(cfg)

    def setup_hmr(self):
        self.HMAR = HMR2023TextureSampler(self.cfg)

    def get_detections(self, image, frame_name, t_, additional_data=None, measurments=None):
        (
            pred_bbox, pred_bbox, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        ) = super().get_detections(image, frame_name, t_, additional_data, measurments)

        # Pad bounding boxes 
        pred_bbox_padded = expand_bbox_to_aspect_ratio(pred_bbox, self.cfg.expand_bbox_shape)

        return (
            pred_bbox, pred_bbox_padded, pred_scores, pred_classes,
            ground_truth_track_id, ground_truth_annotations
        )


@dataclass
class Human4DConfig(FullConfig):
    # override defaults if needed
    expand_bbox_shape: Optional[Tuple[int]] = (192, 256)
    pass


cs = ConfigStore.instance()
cs.store(name="config", node=Human4DConfig)


def extract(cfg, source_path):
    video_name = source_path.split('/')[-1].split('.')[0]
    os.system("rm -rf \"" + cfg.video.output_dir + "/_DEMO/" + video_name + "\"")
    os.makedirs(cfg.video.output_dir + "/_DEMO/" + video_name, exist_ok=True)

    fe = FrameExtractor(source_path)
    log.info('Number of frames: ' + str(fe.n_frames))
    fe.extract_frames(every_x_frame=1, img_name='',
                      dest_path=cfg.video.output_dir + "/_DEMO/" + video_name,
                      start_frame=cfg.video.start_frame, end_frame=cfg.video.end_frame)
    return cfg.video.output_dir + "/_DEMO/" + video_name


def download_task(cfg, queue):
    try:
        client = storage.Client()
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        assert bucket_name
        bucket = client.get_bucket(bucket_name)
        example_video = "kinetic/val/high jump/7pAnkFrScT4_000171_000181.mp4"
        local_video = "tmp_input/" + example_video.split('/')[-1]
        bucket.get_blob(example_video).download_to_filename(local_video)
        source_dir = extract(cfg, local_video)
        print(f"Download {example_video}...")
        queue.put((example_video, source_dir, local_video))
        failed = 0
        max_failed = 5
        while True:
            try:
                ret = requests.post(TASK_SERVER + "/get_task", data={"splash_auth_token": SPLASH_AUTH_TOKEN})
                assert ret.status_code == 200
                task_id = ret.json()["task_id"]
                local_video = "tmp_input/" + task_id.split('/')[-1]
            except Exception:
                print("Failed to get task")
                failed += 1
                if failed >= max_failed:
                    break
                time.sleep(1)
                continue
            bucket.get_blob(task_id).download_to_filename(local_video)
            source_dir = extract(cfg, local_video)
            print(f"Download {task_id}...")
            queue.put((task_id, source_dir, local_video))
    finally:
        queue.put(None)


@torch.no_grad()
def postprocess(final_visuals_dic, video_name):
    persons = defaultdict(lambda: defaultdict(list))
    for k, v in sorted(final_visuals_dic.items(), key=lambda x: x[0]):
        for tracked_time, tid, _3d_joints, smpl, camera, bbox, size in \
                zip(v['tracked_time'], v['tid'], v['3d_joints'],
                    v['smpl'], v['camera'], v['bbox'], v['size']):
            if tracked_time != 0:
                continue
            persons[tid]['3d_joints'].append(_3d_joints)
            persons[tid]['smpl'].append(smpl)
            persons[tid]['camera'].append(camera)
            persons[tid]['bbox'].append(bbox)
            persons[tid]['size'].append(size)
            persons[tid]['time'].append(v['time'])
    wanted = []
    for tid, person in persons.items():
        if len(person['time']) < 64:
            continue
        var = np.stack(person['3d_joints']).std(0).mean()
        bbox = np.stack(person['bbox'])[:, 2:]
        size = np.float32(person['size'])
        relative_area = ((bbox[:, 0] * bbox[:, 1]) / (size[:, 0] * size[:, 1] + 1e-7)).mean()
        absolute_area = (bbox[:, 0] * bbox[:, 1]).mean()
        camera = np.stack(person['camera'])  # s,3
        matrix = np.stack(
            [np.concatenate((x['global_orient'], x['body_pose']), axis=0) for x in person['smpl']])  # s,24,3,3
        rotations = geometry.matrix_to_axis_angle(torch.from_numpy(matrix)).numpy()  # s,24,3
        binascii.b2a_base64(
            rotations.flatten().astype(np.float32).tobytes()).decode(
            "utf-8"),
        new_person = {
            'time': person['time'],
            'var': var.item(),
            'relative_area': relative_area.item(),
            'absolute_area': absolute_area.item(),
            'camera': binascii.b2a_base64(
                camera.flatten().astype(np.float32).tobytes()).decode(
                "utf-8"),
            'rotations': binascii.b2a_base64(
                rotations.flatten().astype(np.float32).tobytes()).decode(
                "utf-8"),
            'dtype': "float32",
        }
        wanted.append(new_person)
    if wanted:
        with open(f"tmp_output/{video_name}.json", "w") as f:
            json.dump(wanted, f, indent=4)


@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> Optional[float]:
    os.makedirs("tmp_input", exist_ok=True)
    os.makedirs("tmp_output", exist_ok=True)
    """Main function for running the PHALP tracker."""
    endpoint = os.getenv("ENDPOINT") + ":" + os.getenv("CUDA_VISIBLE_DEVICES", "0")
    assert endpoint
    queue = multiprocessing.Queue(maxsize=2)
    downloader = multiprocessing.Process(target=download_task, args=(cfg, queue))
    downloader.start()
    phalp_tracker = HMR2_4dhuman(cfg)
    example_task = queue.get()
    assert example_task
    example_video, source_dir, local_video = example_task
    final_visuals_dic = phalp_tracker.track(source_dir)
    assert final_visuals_dic
    postprocess(final_visuals_dic, local_video.split('/')[-1].split('.')[0])
    os.system("rm -rf \"" + source_dir + "\"")
    os.system("rm -rf \"" + local_video + "\"")
    while True:
        task = queue.get()
        if task is None:
            break
        task_id, source_dir, local_video = task
        try:
            final_visuals_dic = phalp_tracker.track(source_dir)
            assert final_visuals_dic
            postprocess(final_visuals_dic, local_video.split('/')[-1].split('.')[0])
        except Exception:
            import traceback
            traceback.print_exc()
        os.system("rm -rf \"" + source_dir + "\"")
        os.system("rm -rf \"" + local_video + "\"")
        try:
            requests.post(TASK_SERVER + "/complete_task", params={"task_id": task_id, "endpoint": endpoint},
                          data={"splash_auth_token": SPLASH_AUTH_TOKEN})
        except Exception:
            pass
        print(f"Process {task_id}...")
    downloader.join()


if __name__ == "__main__":
    main()
