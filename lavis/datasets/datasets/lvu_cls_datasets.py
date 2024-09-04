"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from lavis.datasets.datasets.video_vqa_datasets import VideoQADataset


class LVUCLSDataset(VideoQADataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_paths,
        history,
        num_frames,
        task,
        stride=10,
        split="train",
    ):
        """
        vis_root (string): Root directory of videos (e.g. LVU/videos/)
        ann_root (string): directory to store the gt_dict file
        """
        self.vis_root = vis_root

        task_list = [
            "director",
            "genre",
            "relationship",
            "scene",
            "way_speaking",
            "writer",
            "year",
        ]
        assert task in task_list, f"Invalid task {task}, must be one of {task_list}"
        self.task = task

        if self.task == "director":
            history = 100

        self.gt_dict = {}
        for ann_path in ann_paths:
            self.gt_dict.update(json.load(open(ann_path)))

        self.fps = 10
        self.annotation = {}
        self.stride = stride
        for video_id in self.gt_dict:
            video_dir = os.path.join(self.vis_root, video_id)
            if os.path.exists(video_dir):
                if task in self.gt_dict[video_id]:
                    duration = self.gt_dict[video_id]["duration"]
                    video_length = self.gt_dict[video_id]["num_frames"]
                    label = self.gt_dict[video_id][task]
                    label_after_process = text_processor(label)
                    assert label == label_after_process, "{} not equal to {}".format(
                        label, label_after_process
                    )
                    self.annotation[f"{video_id}_0"] = {
                        "video_id": video_id,
                        "start": 0,
                        "label": label_after_process,
                        "duration": duration,
                        "video_length": video_length,
                        "answer": self.gt_dict[video_id][f"{task}_answer"],
                    }
                    for start in range(
                        self.stride, duration - history + 1, self.stride
                    ):
                        video_start_id = f"{video_id}_{start}"
                        self.annotation[video_start_id] = {
                            "video_id": video_id,
                            "start": start,
                            "label": label_after_process,
                            "duration": duration,
                            "video_length": video_length,
                            "answer": self.gt_dict[video_id][f"{task}_answer"],
                        }
                    self.data_list = list(self.annotation.keys())
            else:
                print(
                    f"Warning: Video directory {video_dir} does not exist or contains no frames."
                )

        # self.data_list = list(self.annotation.keys())
        self.data_list.sort()

        self.history = history
        self.num_frames = num_frames
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    def __getitem__(self, index):
        video_start_id = self.data_list[index]

        start_time = self.annotation[video_start_id]["start"]
        end_time = min(
            self.annotation[video_start_id]["start"] + self.history - 1,
            self.annotation[video_start_id]["duration"],
        )

        start_frame_index = int(start_time * self.fps)
        end_frame_index = min(
            int(end_time * self.fps),
            self.annotation[video_start_id]["video_length"] - 1,
        )
        selected_frame_index = (
            np.rint(np.linspace(start_frame_index, end_frame_index, self.num_frames))
            .astype(int)
            .tolist()
        )

        frame_list = []
        video_dir = os.path.join(
            self.vis_root, self.annotation[video_start_id]["video_id"]
        )
        if os.path.exists(video_dir):
            for frame_index in selected_frame_index:
                frame_path = os.path.join(
                    video_dir, "frame{:06d}.jpg".format(frame_index + 1)
                )
                if os.path.exists(frame_path):
                    frame = Image.open(frame_path).convert("RGB")
                    frame = pil_to_tensor(frame).to(torch.float32)
                    frame_list.append(frame)
                else:
                    print(f"Warning: Frame {frame_path} does not exist.")
            if frame_list:
                # Just in case we don't have enough frames, repeat the last frame
                while len(frame_list) < self.num_frames:
                    frame_list.append(frame_list[-1])
                video = torch.stack(frame_list, dim=1)
                video = self.vis_processor(video)
            else:
                print(f"Warning: No frames found for video {video_dir}.")
                video = None  # or some default value
        else:
            print(f"Warning: Video directory {video_dir} does not exist.")
            video = None  # or some default value

        text_input = self.text_processor(f"what is the {self.task} of the movie?")
        caption = self.text_processor(self.annotation[video_start_id]["label"])
        return {
            "image": video,
            "text_input": text_input,
            "text_output": caption,
            "image_id": video_start_id,
            "question_id": video_start_id,
        }

    def __len__(self):
        return len(self.data_list)


class LVUCLSEvalDataset(LVUCLSDataset):
    def __init__(
        self,
        vis_processor,
        text_processor,
        vis_root,
        ann_paths,
        history,
        num_frames,
        task,
        stride=10,
        split="val",
    ):

        super().__init__(
            vis_processor,
            text_processor,
            vis_root,
            ann_paths,
            history,
            num_frames,
            task,
            stride=stride,
            split=split,
        )

    def __getitem__(self, index):
        video_start_id = self.data_list[index]

        start_time = self.annotation[video_start_id]["start"]
        end_time = min(
            self.annotation[video_start_id]["start"] + self.history - 1,
            self.annotation[video_start_id]["duration"],
        )

        start_frame_index = int(start_time * self.fps)
        end_frame_index = min(
            int(end_time * self.fps),
            self.annotation[video_start_id]["video_length"] - 1,
        )
        selected_frame_index = (
            np.rint(np.linspace(start_frame_index, end_frame_index, self.num_frames))
            .astype(int)
            .tolist()
        )
        # print(start_frame_index, end_frame_index, selected_frame_index, start_time, end_time)
        frame_list = []
        for frame_index in selected_frame_index:
            frame = Image.open(
                os.path.join(
                    self.vis_root,
                    self.annotation[video_start_id]["video_id"],
                    "frame{:06d}.jpg".format(frame_index + 1),
                )
            ).convert("RGB")
            frame = pil_to_tensor(frame).to(torch.float32)
            frame_list.append(frame)
        video = torch.stack(frame_list, dim=1)
        video = self.vis_processor(video)

        text_input = self.text_processor(f"what is the {self.task} of the movie?")
        caption = self.text_processor(self.annotation[video_start_id]["label"])
        return {
            "image": video,
            "text_input": text_input,
            "prompt": text_input,
            "text_output": caption,
            "image_id": video_start_id,
            "question_id": video_start_id,
        }
