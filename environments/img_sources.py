# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import cv2
import glob
import os
import skvideo.io
import random
import tqdm


class BackgroundMatting(object):
    """
    Produce a mask by masking the given color. This is a simple strategy
    but effective for many games.
    """

    def __init__(self, color):
        """
        Args:
            color: a (r, g, b) tuple or single value for grayscale
        """
        self._color = color

    def get_mask(self, img):
        return img == self._color


class ImageSource(object):
    """
    Source of natural images to be added to a simulated environment.
    """

    def get_image(self):
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def reset(self):
        """Called when an episode ends."""
        pass


class FixedColorSource(ImageSource):
    def __init__(self, shape, color):
        """
        Args:
            shape: [h, w]
            color: a 3-tuple
        """
        self.arr = np.zeros((shape[0], shape[1], 3))
        self.arr[:, :] = color

    def get_image(self):
        return self.arr


class RandomColorSource(ImageSource):
    def __init__(self, shape):
        """
        Args:
            shape: [h, w]
        """
        self.shape = shape
        self.arr = None
        self.reset()

    def reset(self):
        self._color = np.random.randint(0, 256, size=(3,))
        self.arr = np.zeros((self.shape[0], self.shape[1], 3))
        self.arr[:, :] = self._color

    def get_image(self):
        return self.arr


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=255):
        """
        Args:
            shape: [h, w]
            strength (int): the strength of noise, in range [0, 255]
        """
        self.shape = shape
        self.strength = strength

    def get_image(self):
        return np.random.randn(self.shape[0], self.shape[1], 3) * self.strength


class RandomImageSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of image files
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        self.total_frames = (
            self.total_frames if self.total_frames else len(self.filelist)
        )
        self.arr = np.zeros(
            (self.total_frames, self.shape[0], self.shape[1])
            + ((3,) if not self.grayscale else (1,))
        )
        for i in range(self.total_frames):
            # if i % len(self.filelist) == 0: random.shuffle(self.filelist)
            fname = self.filelist[i % len(self.filelist)]
            if self.grayscale:
                im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
                self.arr[i] = cv2.resize(
                    im, (self.shape[1], self.shape[0])
                )[..., None]  ## THIS IS NOT A BUG! cv2 uses (width, height)
            else:
                im = cv2.imread(fname, cv2.IMREAD_COLOR)
                self.arr[i] = cv2.resize(
                    im, (self.shape[1], self.shape[0])
                )  ## THIS IS NOT A BUG! cv2 uses (width, height)

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        return self.arr[self._loc]


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    # def build_arr(self):
    #     if not self.total_frames:
    #         self.total_frames = 0
    #         self.arr = None
    #         random.shuffle(self.filelist)
    #         for fname in tqdm.tqdm(
    #             self.filelist, desc="Loading videos for natural", position=0
    #         ):
    #             if self.grayscale:
    #                 frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
    #             else:
    #                 frames = skvideo.io.vread(fname)
    #             local_arr = np.zeros(
    #                 (frames.shape[0], self.shape[0], self.shape[1])
    #                 + ((3,) if not self.grayscale else (1,))
    #             )
    #             for i in tqdm.tqdm(
    #                 range(frames.shape[0]), desc="video frames", position=1
    #             ):
    #                 local_arr[i] = cv2.resize(
    #                     frames[i], (self.shape[1], self.shape[0])
    #                 )  ## THIS IS NOT A BUG! cv2 uses (width, height)
    #             if self.arr is None:
    #                 self.arr = local_arr
    #             else:
    #                 self.arr = np.concatenate([self.arr, local_arr], 0)
    #             self.total_frames += local_arr.shape[0]
    #     else:
    #         self.arr = np.zeros(
    #             (self.total_frames, self.shape[0], self.shape[1])
    #             + ((3,) if not self.grayscale else (1,))
    #         )
    #         total_frame_i = 0
    #         file_i = 0
    #         with tqdm.tqdm(
    #             total=self.total_frames, desc="Loading videos for natural"
    #         ) as pbar:
    #             while total_frame_i < self.total_frames:
    #                 if file_i % len(self.filelist) == 0:
    #                     random.shuffle(self.filelist)
    #                 file_i += 1
    #                 fname = self.filelist[file_i % len(self.filelist)]
    #                 if self.grayscale:
    #                     frames = skvideo.io.vread(
    #                         fname, outputdict={"-pix_fmt": "gray"}
    #                     )
    #                 else:
    #                     frames = skvideo.io.vread(fname)
    #                 for frame_i in range(frames.shape[0]):
    #                     if total_frame_i >= self.total_frames:
    #                         break
    #                     if self.grayscale:
    #                         self.arr[total_frame_i] = cv2.resize(
    #                             frames[frame_i], (self.shape[1], self.shape[0])
    #                         )[
    #                             ..., None
    #                         ]  ## THIS IS NOT A BUG! cv2 uses (width, height)
    #                     else:
    #                         self.arr[total_frame_i] = cv2.resize(
    #                             frames[frame_i], (self.shape[1], self.shape[0])
    #                         )
    #                     pbar.update(1)
    #                     total_frame_i += 1
    
    def build_arr(self):
        if not self.total_frames:
            # Strategy A: Load all videos
            self.total_frames = 0
            self.arr = None
            random.shuffle(self.filelist)
            video_count = 0  # ADD THIS
            print(f"Loading ALL videos from {len(self.filelist)} files...")  # ADD THIS
            
            for fname in tqdm.tqdm(self.filelist, desc="Loading videos for natural", position=0):
                video_count += 1  # ADD THIS
                if self.grayscale: frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                else:              frames = skvideo.io.vread(fname)
                local_arr = np.zeros((frames.shape[0], self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
                for i in tqdm.tqdm(range(frames.shape[0]), desc="video frames", position=1):
                    local_arr[i] = cv2.resize(frames[i], (self.shape[1], self.shape[0])) ## THIS IS NOT A BUG! cv2 uses (width, height)
                if self.arr is None:
                    self.arr = local_arr
                else:
                    self.arr = np.concatenate([self.arr, local_arr], 0)
                self.total_frames += local_arr.shape[0]
                
            print(f"Strategy A: Loaded {video_count} videos, total {self.total_frames} frames")  # ADD THIS
            
        else:
            # Strategy B: Load until reaching total_frames
            self.arr = np.zeros((self.total_frames, self.shape[0], self.shape[1]) + ((3,) if not self.grayscale else (1,)))
            total_frame_i = 0
            file_i = 0
            video_count = 0  # ADD THIS
            loaded_videos = set()  # ADD THIS - track unique videos
            
            print(f"Loading videos until {self.total_frames} total frames from {len(self.filelist)} available files...")  # ADD THIS
            
            with tqdm.tqdm(total=self.total_frames, desc="Loading videos for natural") as pbar:
                while total_frame_i < self.total_frames:
                    if file_i % len(self.filelist) == 0: 
                        random.shuffle(self.filelist)
                        if file_i > 0:  # ADD THIS - don't print on first iteration
                            print(f"  Cycled through all {len(self.filelist)} files, reshuffling...")  # ADD THIS
                    
                    file_i += 1
                    fname = self.filelist[file_i % len(self.filelist)]
                    
                    # Track video loading  # ADD THIS BLOCK
                    if fname not in loaded_videos:
                        loaded_videos.add(fname)
                        video_count += 1
                        print(f"  Loading video {video_count}: {os.path.basename(fname)}")
                    else:
                        print(f"  Re-loading video: {os.path.basename(fname)} (already loaded)")
                    
                    if self.grayscale: frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                    else:              frames = skvideo.io.vread(fname)
                    
                    frames_from_this_video = 0  # ADD THIS - track frames per video
                    for frame_i in range(frames.shape[0]):
                        if total_frame_i >= self.total_frames: break
                        if self.grayscale:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0]))[..., None]
                        else:
                            self.arr[total_frame_i] = cv2.resize(frames[frame_i], (self.shape[1], self.shape[0]), )
                        pbar.update(1)
                        total_frame_i += 1
                        frames_from_this_video += 1  # ADD THIS
                    
                    print(f"    Used {frames_from_this_video}/{frames.shape[0]} frames from this video")  # ADD THIS
            
            print(f"Strategy B: Loaded {len(loaded_videos)} unique videos (some possibly multiple times)")  # ADD THIS
            print(f"Total video file accesses: {file_i}")  # ADD THIS
            print(f"Final frame count: {total_frame_i}")  # ADD THIS
            
            # Print summary of loaded videos  # ADD THIS BLOCK
            print("Loaded video files:")
            for i, fname in enumerate(sorted(loaded_videos)):
                print(f"  {i+1}. {os.path.basename(fname)}")

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        img = self.arr[self._loc % self.total_frames]
        self._loc += 1
        return img


def make_img_source(src_type, img_shape, resource_files, total_frames, grayscale):
    if src_type == "color":
        img_source = RandomColorSource(img_shape)
    elif src_type == "noise":
        img_source = NoiseSource(img_shape)
    else:
        files = glob.glob(os.path.expanduser(resource_files))
        assert len(files), f"Pattern {resource_files} does not match any files"
        if src_type == "images":
            img_source = RandomImageSource(
                img_shape, files, total_frames=total_frames, grayscale=grayscale
            )
        elif src_type == "video":
            img_source = RandomVideoSource(
                img_shape, files, total_frames=total_frames, grayscale=grayscale
            )
        else:
            raise Exception(f"img_source {src_type} not defined.")
    return img_source
