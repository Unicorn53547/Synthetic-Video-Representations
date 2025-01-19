import os
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE
from ssv2 import SSVideoClsDataset
from dead_leaves import VideoMAE_on_the_fly
from dataset_mixed import VideoMAE_on_the_fly_mixed, AlignedDataset, img2video
from torchvision import datasets, transforms




    
class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        rotate=args.train_rotate,
        translate=args.train_translate,
        mix_all=args.mix_all,
        video_perturbe=args.train_perturbe,
        sev=args.severity,)
    print("Data Aug = %s" % str(transform))
    return dataset

def build_synthetic_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE_on_the_fly(
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        #TODO: add synthetics args
    )
    print("Data Aug = %s" % str(transform))
    return dataset


def build_mixed_dataset(args):
    transform = DataAugmentationForVideoMAE(args)

    
    video_dataset = VideoMAE_on_the_fly_mixed(
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        shape_mode=args.shape,
        resolution=256,
        sigma=3,
        virtual_dataset_size=args.virtual_dataset_size,
        min_duration=100,
        max_duration=200,
        max_iters=args.max_iters,
        acceleration=args.acc,
        texture_folder='',
        textured=args.textured,
        num_textures=10000,
        affine_transform=args.affine,
        static = args.static,
        rotate_only=args.rotate_only,
        max_acc=args.max_acc,
        min_speed=args.min_speed,
        max_speed =args.max_speed, 
    )
    

    if args.img_data_type == 'stylegan':
        img_dataset = datasets.ImageFolder(root=args.img_data_path, transform=None)
    elif args.img_data_type == 'imagenet':
        img_dataset = datasets.ImageFolder(root=args.img_data_path, transform=None)

    print('Original Image dataset size: {}'.format(len(img_dataset)))
    
    #log how much train data be replaced with image
    print('Replacing {}% of dataset with {}% of image'.format(args.mixed_ratio * 100, args.use_real_ratio * 100))
    dataset = AlignedDataset(video_dataset, img_dataset, args.use_real_ratio, repetitions=args.num_frames, mixed_ratio=args.mixed_ratio, transform=transform)
    

    return dataset

class AlignedDataset_video(torch.utils.data.Dataset):
    def __init__(self, syn_dataset, real_dataset, mixed_ratio=0.5):
        self.syn_dataset = syn_dataset
        self.real_dataset = real_dataset

        self.real_data_len = len(self.real_dataset)
        self.video_len = int(len(self.real_dataset) * (1 - mixed_ratio))


    def __len__(self):
        return len(self.real_dataset)

    def __getitem__(self, index):
        if index < self.video_len:
            idx = np.random.randint(0, self.real_data_len)
            process_data, mask =  self.syn_dataset[idx]
        else:
            #randomly select an image from imagenet
            idx = np.random.randint(0, self.real_data_len)
            process_data, mask =  self.real_dataset[idx]
            
        return process_data, mask
    
def build_mixed_dataset_v3(args):
    transform = DataAugmentationForVideoMAE(args)

    syn_video_dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False,
        video_perturbe=False,
        sev=args.severity,)

    print('Replacing {}% of dataset with real video'.format(args.mixed_ratio * 100))
    real_video_dataset = VideoMAE(
        root=None,
        setting=args.img_data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    
    dataset = AlignedDataset_video(syn_video_dataset, real_video_dataset, args.mixed_ratio)

    print(dataset)

    return dataset

def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
            severity=args.severity,
            video_perturbe=args.video_perturbe)
        nb_classes = 400
    
    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            video_perturbe=args.video_perturbe,
            args=args,
            severity=args.severity)
        nb_classes = 101
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
