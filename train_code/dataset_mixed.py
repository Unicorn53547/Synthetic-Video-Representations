import os
import random

import torch

import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import decord


def cv2_resize(image, target_height_width, interpolation=cv2.INTER_NEAREST):
  if len(image.shape) == 2:
    return cv2.resize(image, target_height_width[::-1], interpolation=interpolation)
  else:
    return cv2.resize(image.transpose((1, 2, 0)), target_height_width[::-1], interpolation=interpolation).transpose((2, 0, 1))

def cv2_imread(file, return_BGR=False):
  im = cv2.imread(file)
  if im is None:
    raise Exception('Image {} could not be read!'.format(file))
  im = im.transpose(2,0,1)
  if return_BGR:
    return im
  return im[::-1, :, :]

def listdir(folder, prepend_folder=False, extension=None, type=None):
  assert type in [None, 'file', 'folder'], "Type must be None, 'file' or 'folder'"
  files = [k for k in os.listdir(folder) if (True if extension is None else k.endswith(extension))]
  if type == 'folder':
    files = [k for k in files if os.path.isdir(folder + '/' + k)]
  elif type == 'file':
    files = [k for k in files if not os.path.isdir(folder + '/' + k)]
  if prepend_folder:
    files = [folder + '/' + f for f in files]
  return files

class VideoMAE_on_the_fly_mixed(torch.utils.data.Dataset):
    """Load your own video classification dataset.
    """
    def __init__(self,
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 lazy_init=False,
                 temporal_jitter=False,
                 shape_mode='circle',
                 resolution=256,
                 sigma=3,
                 virtual_dataset_size=9537,
                 min_duration=100,
                 max_duration=200,
                 max_iters=500,
                 acceleration=False,
                 texture_folder='',
                 textured=False,
                 num_textures=10000,
                 affine_transform=False,
                 static=False,
                 rotate_only=False,
                 max_acc=0.03,
                 min_speed=1.2,
                 max_speed=1.5,                # speed_range=[1.2, 1.5],
                 ):

        super(VideoMAE_on_the_fly_mixed, self).__init__()
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.shape_mode = shape_mode
        self.resolution = resolution
        self.sigma = sigma
        self.max_iters = max_iters
        self.virtual_dataset_size = virtual_dataset_size
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.acc = acceleration
        self.affine = affine_transform
        self.num_textures = num_textures
        self.static = static
        self.rotate_only = rotate_only
        self.max_acc = max_acc
        self.min_speed=min_speed
        self.max_speed=max_speed
        self.transform = transform
        self.lazy_init = lazy_init
        
        # global variables for generated samplesd
        self.reset_buffers()

        self.textured = False
        if self.textured:
            assert not texture_folder is None
            self.texture_folder = texture_folder
            self.load_texture()


    def load_texture(self):
        textures_wmm_folders = sorted(listdir(self.texture_folder, prepend_folder=True))
        texture_files = []
        print("listing textures wmm")
        n_textures_to_load = self.num_textures
        for dir in tqdm(textures_wmm_folders):
            texture_files.extend(listdir(dir, prepend_folder=True))
        print("End listing textures wmm")

        assert len(texture_files) >= n_textures_to_load
        texture_files = random.sample(texture_files, n_textures_to_load)
        loaded_textures = []

        print("Loading textures to memory!")
        for texture_f in tqdm(texture_files):
            loaded_textures.append(cv2_resize(cv2_imread(texture_f), (self.resolution * 2, self.resolution * 2)))
        print("Ended textures to memory!")
        self.loaded_textures = loaded_textures
    
    def __getitem__(self, index):
        #get a random val from (min_duration, max_duration)
        duration = random.randint(self.min_duration, self.max_duration)
        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images = self.generate_dead_leaves(duration, segment_indices, skip_offsets)
      
        self.reset_buffers()

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        
    
        return process_data, mask

    def __len__(self):
        return self.virtual_dataset_size

    def _sample_train_indices(self, num_frames):
        average_duration = (num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets
    

    def generate_dead_leaves(self, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        self.init_buffer()

        sampled_list = [Image.fromarray(self.dead_leaves(frame_id_list[i])).convert('RGB') for i in range(len(frame_id_list))]

        return sampled_list

    def reset_buffers(self):
        self.centers_buffer = []
        self.speed_buffer = []
        self.direction_buffer = []
        self.radius_buffer = []
        self.colors_buffer = []
        self.shape_buffer = []
        #oriented_square only
        self.theta_buffer = [1] * self.max_iters
        #rectangle only
        self.rec_buffer = [1] * self.max_iters
        #
        self.angle_buffer = [1] * self.max_iters
        self.acceleration_buffer = []
        #affine transform
        self.angle_speed_buffer = []
        self.scale_x_speed_buffer = []
        self.scale_y_speed_buffer = []
        self.shear_x_speed_buffer = []
        self.shear_y_speed_buffer = []

        self.texture_id_buffer = []


    def init_buffer(self):
        res = self.resolution
        sigma = self.sigma
        max_iters = self.max_iters
        shape_mode = self.shape_mode

        rmin = 0.1
        rmax = 1

        # compute distribution of radiis (exponential distribution with lambda = sigma):
        k = 200
        r_list = np.linspace(rmin, rmax, k)
        r_dist = 1./(r_list ** sigma)
        if sigma > 0:
            # normalize so that the tail is 0 (p(r >= rmax)) = 0
            r_dist = r_dist - 1/rmax**sigma
        r_dist = np.cumsum(r_dist)
        # normalize so that cumsum is 1.
        r_dist = r_dist/r_dist.max()
        
        for i in range(max_iters):
            available_shapes = ['circle', 'square', 'oriented_square','rectangle', 'triangle', 'quadrilater']
            assert shape_mode in available_shapes or shape_mode == 'mixed'
            if shape_mode == 'mixed':
                shape = random.choice(available_shapes)
                self.shape_buffer.append(shape)
            else:
                shape = shape_mode

            color = tuple([int(k) for k in np.random.uniform(0, 1, 3) * 255])
            self.colors_buffer.append(color)

            # radius
            r_p = np.random.uniform(0,1)
            r_i = np.argmin(np.abs(r_dist - r_p))
            radius = max(int(r_list[r_i] * res), 1)
            self.radius_buffer.append(radius)

            #center
            center_x, center_y = np.array(np.random.uniform(0,res*2, size=2),dtype='int32')
            self.centers_buffer.append((center_x, center_y))

            # direction: random angle in [0, 2pi]
            dir = np.random.uniform(0, 2*np.pi, 1)
            self.direction_buffer.append(dir)

            # speed: random speed in [1.2, 1.5]  To be realistic under 25 fps (ucf101)
            speed = np.random.uniform(self.min_speed,self.max_speed, 1)
            self.speed_buffer.append(speed)

            #get a random acceleration for each object
            if self.acc:
                acceleration = np.random.uniform(-self.max_acc, self.max_acc, 1)
                self.acceleration_buffer.append(acceleration)
            
            if self.textured:
                texture_id = np.random.randint(0, len(self.loaded_textures) - 1)
                self.texture_id_buffer.append(texture_id)

            if shape == 'circle':
                if self.affine:
                    angle_speed = np.random.uniform(-1/100*np.pi, 1/100*np.pi)
                    scale_x_speed = np.random.uniform(-0.005, 0.005)
                    scale_y_speed = np.random.uniform(-0.005, 0.005)
                    shear_x_speed = np.random.uniform(-0.005, 0.005)
                    shear_y_speed = np.random.uniform(-0.005, 0.005)
                    self.angle_speed_buffer.append(angle_speed)
                    self.scale_x_speed_buffer.append(scale_x_speed)
                    self.scale_y_speed_buffer.append(scale_y_speed)
                    self.shear_x_speed_buffer.append(shear_x_speed)
                    self.shear_y_speed_buffer.append(shear_y_speed)
            else:
                if self.affine:
                    angle_speed = np.random.uniform(-1/100*np.pi, 1/100*np.pi)
                    scale_x_speed = np.random.uniform(-0.005, 0.005)
                    scale_y_speed = np.random.uniform(-0.005, 0.005)
                    shear_x_speed = np.random.uniform(-0.005, 0.005)
                    shear_y_speed = np.random.uniform(-0.005, 0.005)
                    self.angle_speed_buffer.append(angle_speed)
                    self.scale_x_speed_buffer.append(scale_x_speed)
                    self.scale_y_speed_buffer.append(scale_y_speed)
                    self.shear_x_speed_buffer.append(shear_x_speed)
                    self.shear_y_speed_buffer.append(shear_y_speed)
                if shape == 'oriented_square' :
                    theta = np.random.uniform(0, 2 * np.pi)
                    self.theta_buffer[i] = theta
                        
                elif shape == 'rectangle':
                    a = np.random.uniform(0, 0.5*np.pi, 1)
                    self.rec_buffer[i] = a
                else:
                    angles = sorted(np.random.uniform(0, 2*np.pi, 3 if shape == 'triangle' else 4))
                    self.angle_buffer[i] = angles
    
    def dead_leaves(self, frame=0):
        res = self.resolution
        sigma = self.sigma
        max_iters = self.max_iters
        shape_mode = self.shape_mode

        img = np.zeros((res, res, 3), dtype=np.uint8)
      
        for i in range(max_iters):
            available_shapes = ['circle', 'square', 'oriented_square','rectangle', 'triangle', 'quadrilater']
            assert shape_mode in available_shapes or shape_mode == 'mixed'
            if shape_mode == 'mixed':
                shape = self.shape_buffer[i]
            else:
                shape = shape_mode

            # else:
            color = self.colors_buffer[i]
            radius = self.radius_buffer[i]
            center_x, center_y = self.centers_buffer[i]
            dir = self.direction_buffer[i]
            v = self.speed_buffer[i]

            #add acceleration if needed
            if self.acc:
                a = self.acceleration_buffer[i]
                v = v + a * frame
            if self.static:
                v = 0
            #update center and rewrite buffer
            update_x = center_x + v * np.cos(dir) * frame
            update_y = center_y + v * np.sin(dir) * frame
            center_x, center_y = np.array((update_x[0], update_y[0]), dtype='int32')

            if self.textured:
                texture_id = self.texture_id_buffer[i]
                texture_f = self.loaded_textures[texture_id].transpose((1,2,0))
                # mask_center_x, mask_center_y = res, res

            if center_x <= -res or center_x >= 2*res or center_y <= -res or center_y >= 2*res:
                    continue
            
            if shape == 'circle':
                #use cv2.ellipse to draw a circle
                radius_x = radius
                radius_y = radius
                rotate_angle = 0
                if self.affine:
                    angle_speed = self.angle_speed_buffer[i]
                    rotate_angle = angle_speed * frame
                    #scale
                    #TODO  can not add texture to scaled circlr yet
                    if not self.rotate_only or not self.textured:
                        scale_x_speed = self.scale_x_speed_buffer[i]
                        scale_y_speed = self.scale_y_speed_buffer[i]
                        scale_x = 1 + scale_x_speed * frame
                        scale_y = 1 + scale_y_speed * frame
                        radius_x = radius * scale_x
                        radius_y = radius * scale_y
                        #no shear for circle
                img = cv2.ellipse(img, (center_x, center_y), (int(radius_x), int(radius_y)), rotate_angle, 0, 360, color, -1)
                if self.textured:
                    blank = np.zeros((res, res), dtype=np.uint8)                   
                    mask = cv2.ellipse(blank, (center_x, center_y), (radius, radius), 0, 0, 360, 255, -1)

                    ori_corners = np.array(((0, radius), (radius, 0), (2 * radius, radius)), dtype='float32')
                    c = np.cos(rotate_angle)
                    s = np.sin(rotate_angle)
                    new_corners = np.array(((center_x - radius_x * c, center_y + radius_x * s), (center_x - radius_y * s, center_y - radius_y * c), (center_x + radius * c, center_y - radius * s)), dtype='float32')
                    affine_matrix = cv2.getAffineTransform(ori_corners, new_corners)
                    new_texture = cv2.warpAffine(texture_f, affine_matrix, (res, res))
                    masked_texture = cv2.bitwise_and(new_texture, new_texture, mask=mask)
                    img = img + masked_texture
            else:
                if self.affine:
                    affine_c_x, affine_c_y = radius // 2, radius // 2
                    angle_speed = self.angle_speed_buffer[i]
                    rotate_angle = angle_speed * frame
                    c, s = np.cos(rotate_angle), np.sin(rotate_angle)
                    R = np.array(((c, -s, affine_c_x * (1 - c) + affine_c_y * s), (s, c, affine_c_y * (1 - c) - affine_c_x * s), (0, 0, 1)))
                    #scale
                    scale_x_speed = self.scale_x_speed_buffer[i]
                    scale_y_speed = self.scale_y_speed_buffer[i]
                    scale_x = 1 + scale_x_speed * frame
                    scale_y = 1 + scale_y_speed * frame
                    S = np.array(((scale_x, 0, affine_c_y * (1 - scale_x)), (0, scale_y, affine_c_y * (1 - scale_y)), (0, 0, 1)))
                    #shear
                    shear_x_speed = self.shear_x_speed_buffer[i]
                    shear_y_speed = self.shear_y_speed_buffer[i]
                    shear_x = shear_x_speed * frame
                    shear_y = shear_y_speed * frame
                    Sh = np.array(((1, shear_x, -shear_x * affine_c_x), (shear_y, 1, -shear_y * affine_c_y), (0, 0, 1)))
                    #affine transformation matrix
                    A = R @ S @ Sh
                    if self.rotate_only:
                        A = R
                if shape == 'square':
                    side = radius * np.sqrt(2)
                    corners = np.array(((0, 0),
                                    (+ side, 0),
                                    (+ side, +side),
                                    (0, +side)), dtype='int32')
                    
                elif shape == 'oriented_square':
                    side = radius * np.sqrt(2)
                    corners = np.array(((- side / 2, - side / 2),
                                        (+ side / 2, - side / 2),
                                        (+ side / 2, + side / 2),
                                        (- side / 2, + side / 2)), dtype='int32')
                    theta = self.theta_buffer[i]
                    c, s = np.cos(theta), np.sin(theta)
                    R = np.array(((c, -s), (s, c)))
                    corners = (R @ corners.transpose()).transpose()
                    corners = corners + np.array((radius, radius))
                elif shape == 'rectangle':
                    a = self.rec_buffer[i]
                    corners = np.array(((+ radius * np.cos(a) + radius, + radius * np.sin(a) + radius),
                                    (+ radius * np.cos(a) + radius, - radius * np.sin(a) + radius),
                                    (- radius * np.cos(a) + radius, - radius * np.sin(a) + radius),
                                    (- radius * np.cos(a) + radius, + radius * np.sin(a) + radius)), dtype='int32')[:,:,0]

                else:
                    angles = self.angle_buffer[i]
                    corners = []
                    for a in angles:
                        corners.append((radius * np.cos(a) + radius, radius * np.sin(a) + radius))
                    corners = np.array(corners, dtype='int32')
                #apply affine transformation
                ori_corners = corners
                if self.affine:
                    corners_3d = np.concatenate((corners, np.ones((corners.shape[0], 1))), axis=1)
                    corners = (A @ corners_3d.transpose()).transpose()[:, :2]
                
                corners = np.array((center_x, center_y)) + np.array(corners)
                
                img = cv2.fillPoly(img, np.array(corners, dtype='int32')[None], color=color)
                if self.textured:
                    blank = np.zeros((res, res), dtype=np.uint8)
                    mask = cv2.fillPoly(blank, np.array(corners, dtype='int32')[None], 255)
                    affine_matrix = cv2.getAffineTransform(np.float32(ori_corners[:3]), np.float32(corners[:3]))
                    new_texture = cv2.warpAffine(texture_f, affine_matrix, (res, res))
                    masked_texture = cv2.bitwise_and(new_texture, new_texture, mask=mask)
                    img = img + masked_texture
        
        return np.clip(img, 0, 255)


def create_random_subset(dataset, percentage):
    num_examples = len(dataset)
    num_samples = int(num_examples * percentage)

    # Generate random indices
    random_indices = torch.randperm(num_examples)[:num_samples]

    # Select subset based on random indices
    subset = torch.utils.data.Subset(dataset, random_indices)

    return subset

class img2video(torch.utils.data.Dataset):
    def __init__(self,
                 root,
                 setting,
                 train=True,
                 test_mode=False,
                 name_pattern='img_%05d.jpg',
                 video_ext='mp4',
                 is_color=True,
                 modality='rgb',
                 num_segments=1,
                 num_crop=1,
                 new_length=1,
                 new_step=1,
                 transform=None,
                 temporal_jitter=False,
                 video_loader=False,
                 use_decord=False,
                 lazy_init=False):

        super(img2video, self).__init__()
        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init


        if not self.lazy_init:
            self.clips = self._make_dataset(root, setting)
            if len(self.clips) == 0:
                raise(RuntimeError("Found 0 video clips in subfolders of: " + root + "\n"
                                   "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):

        directory, target = self.clips[index]
        
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                video_name = directory
            else:
                video_name = '{}.{}'.format(directory, self.video_ext)

            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)
        #sample one random frame
        indices = random.randint(0, duration - 1)
        image_sample = self._video_TSN_decord_batch_loader(decord_vr, indices)
        #repeat numm_frame times
        images = [image_sample] * self.new_length
        
        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
        
        return process_data, mask

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting):
        if not os.path.exists(setting):
            raise(RuntimeError("Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))
        clips = []
        with open(setting) as split_f:
            data = split_f.readlines()
            for line in data:
                line_info = line.split(' ')
                # line format: video_path, video_duration, video_label
                if len(line_info) < 2:
                    raise(RuntimeError('Video input format is not correct, missing one or more element. %s' % line))
                clip_path = os.path.join(line_info[0])
                target = int(line_info[1])
                item = (clip_path, target)
                clips.append(item)
        return clips


    def _video_TSN_decord_batch_loader(self, video_reader, indices):
        #only get one frame based on indices
        frame_id = indices
        video_data = video_reader.get_batch([frame_id]).asnumpy()
        sampled_list = Image.fromarray(video_data[0, :, :, :]).convert('RGB')
        
        return sampled_list
    
# Custom dataset class for aligning ImageNet with UCF101
class AlignedDataset(torch.utils.data.Dataset):
    def __init__(self, ucf101_dataset, imagenet_dataset, subset_percentage, repetitions=16, mixed_ratio=0.05, transform=None):
        self.ucf101_dataset = ucf101_dataset
        self.imagenet_dataset = create_random_subset(imagenet_dataset, subset_percentage)
        self.repetitions = repetitions
        self.new_length = repetitions
        self.img_data_len = len(self.imagenet_dataset)
        self.video_len = int(len(self.ucf101_dataset) * (1 - mixed_ratio))
        self.img_len = len(self.ucf101_dataset) - self.video_len
        self.transform = transform

    def __len__(self):
        return len(self.ucf101_dataset)

    def __getitem__(self, index):
        if index < self.video_len:
            process_data, mask =  self.ucf101_dataset[index]
        else:
            #randomly select an image from imagenet
            idx = np.random.randint(0, self.img_data_len)
            img_sample = self.imagenet_dataset[idx][0]
            images = [img_sample] * self.repetitions
            process_data, mask = self.transform((images, None)) # T*C,H,W
            
            process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W
            
        return process_data, mask

    



