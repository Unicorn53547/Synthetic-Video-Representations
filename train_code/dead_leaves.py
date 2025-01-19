import numpy as np
import torch
from PIL import Image
import video_transforms as video_transforms 
import volume_transforms as volume_transforms
import random
import cv2


class VideoMAE_on_the_fly(torch.utils.data.Dataset):
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
                 ):

        super(VideoMAE_on_the_fly, self).__init__()

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

        self.transform = transform
        self.lazy_init = lazy_init
        



        # global variables for generated samplesd
        self.centers_buffer = []
        self.speed_buffer = []
        self.direction_buffer = []
        self.radius_buffer = []
        self.colors_buffer = []
        self.shape_buffer = []
        #oriented_square only
        self.theta_buffer = []
        #rectangle only
        self.rec_buffer = []
    
        self.angle_buffer = []

        



    def __getitem__(self, index):

        duration = random.randint(self.min_duration, self.max_duration)
        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images = self.generate_dead_leaves(duration, segment_indices, skip_offsets)
        
        self.reset_buffers()

        process_data, mask = self.transform((images, None)) # T*C,H,W
        process_data = process_data.view((self.new_length, 3) + process_data.size()[-2:]).transpose(0,1)  # T*C,H,W -> T,C,H,W -> C,T,H,W

        return (process_data, mask) 

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
        self.theta_buffer = []
        #rectangle only
        self.rec_buffer = []
        #
        self.angle_buffer = []

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

            #store universal buffer
            # color
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

            # speed: random speed in [0.5, 1.5]
            speed = np.random.uniform(1.2, 1.5, 1)
            self.speed_buffer.append(speed)

            if shape == 'circle':
                continue
            else:
                if shape == 'square' or shape == 'oriented_square':
                    if shape == 'oriented_square':
                        theta = np.random.uniform(0, 2 * np.pi)
                        self.theta_buffer.append(theta)
                elif shape == 'rectangle':
                    # sample one points in the firrst quadrant, and get the two other symmetric
                    a = np.random.uniform(0, 0.5*np.pi, 1)
                    self.rec_buffer.append(a)
                    
    
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

            color = self.colors_buffer[i]
            radius = self.radius_buffer[i]
            center_x, center_y = self.centers_buffer[i]
            dir = self.direction_buffer[i]
            v = self.speed_buffer[i]
            #update center and rewrite buffer
            update_x = center_x + v * np.cos(dir) * frame
            update_y = center_y + v * np.sin(dir) * frame
            center_x, center_y = np.array((update_x[0], update_y[0]), dtype='int32')
            
            if shape == 'circle':
                img = cv2.circle(img, (center_x, center_y),radius=radius, color=color, thickness=-1)
            else:
                if shape == 'square' or shape == 'oriented_square':
                    side = radius * np.sqrt(2)
                    corners = np.array(((- side / 2, - side / 2),
                                        (+ side / 2, - side / 2),
                                        (+ side / 2, + side / 2),
                                        (- side / 2, + side / 2)), dtype='int32')
                    if shape == 'oriented_square':
                        theta = self.theta_buffer[i]
                        c, s = np.cos(theta), np.sin(theta)
                        R = np.array(((c, -s), (s, c)))
                        corners = (R @ corners.transpose()).transpose()
                elif shape == 'rectangle':
                    a = self.rec_buffer[i]
                    corners = np.array(((+ radius * np.cos(a), + radius * np.sin(a)),
                                        (+ radius * np.cos(a), - radius * np.sin(a)),
                                        (- radius * np.cos(a), - radius * np.sin(a)),
                                        (- radius * np.cos(a), + radius * np.sin(a))), dtype='int32')[:,:,0]

                else:
                    angles = self.angle_buffer[i]
                    corners = []
                    for a in angles:
                        corners.append((radius * np.cos(a), radius * np.sin(a)))

                
                corners = np.array((center_x, center_y)) + np.array(corners)
                
                img = cv2.fillPoly(img, np.array(corners, dtype='int32')[None], color=color)
                       
        return np.clip(img, 0, 255)
