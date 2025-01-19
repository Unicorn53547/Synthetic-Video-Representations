import sys, os
from p_tqdm import p_map

sys.path.append('.')
from utils_generation import *

# global variables for generated samplesd
VIDEO_NUM=1000
centers_buffer = [[] for _ in range(VIDEO_NUM)]
speed_buffer = [[] for _ in range(VIDEO_NUM)]
direction_buffer = [[] for _ in range(VIDEO_NUM)]
radius_buffer = [[] for _ in range(VIDEO_NUM)]
colors_buffer = [[] for _ in range(VIDEO_NUM)]
shape_buffer = [[] for _ in range(VIDEO_NUM)]
acc_buffer = [[] for _ in range(VIDEO_NUM)]
#oriented_square only
theta_buffer = [[1]*500 for _ in range(VIDEO_NUM)]
#rectangle only
rec_buffer = [[2]*500 for _ in range(VIDEO_NUM)]
#
angle_buffer = [[3]*500 for _ in range(VIDEO_NUM)]

# global variables for affine transformation
angle_speed_buffer = [[] for _ in range(VIDEO_NUM)]
scale_x_speed_buffer = [[] for _ in range(VIDEO_NUM)]
scale_y_speed_buffer = [[] for _ in range(VIDEO_NUM)]
shear_x_speed_buffer = [[] for _ in range(VIDEO_NUM)]
shear_y_speed_buffer = [[] for _ in range(VIDEO_NUM)]
texture_id_buffer = [[] for _ in range(VIDEO_NUM)]

frame_list = []
color_back = []




# ported from dead_leaves.m
#frame 0, store center/color/radius/direction/speed; frame > 0, update center and fill with all other info
def dead_leaves(res, sigma, shape_mode='mixed', max_iters=2000, textures=None, frame=0, video_index=0):
    #black background
    img = np.zeros((res, res, 3), dtype=np.uint8)
    #video index to 0-999
    video_index = video_index % VIDEO_NUM
    
    if frame == 0:
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
        r_dist = r_dist/r_dist.max()
        
    for i in range(max_iters):
        available_shapes = ['circle', 'square', 'oriented_square','rectangle', 'triangle', 'quadrilater']
        assert shape_mode in available_shapes or shape_mode == 'mixed'
        if shape_mode == 'mixed':
            if frame == 0:
                shape = random.choice(available_shapes)
                shape_buffer[video_index].append(shape)
            else:
                shape = shape_buffer[video_index][i]
        else:
            shape = shape_mode

        #store or load universal buffer
        if frame == 0:
            # color
            if textures is not None:
                color = (0, 0, 0)
                colors_buffer[video_index].append(color)
            else:
                color = tuple([int(k) for k in np.random.uniform(0, 1, 3) * 255])
                colors_buffer[video_index].append(color)

            # radius
            r_p = np.random.uniform(0,1)
            r_i = np.argmin(np.abs(r_dist - r_p))
            radius = max(int(r_list[r_i] * res), 1)
            radius_buffer[video_index].append(radius)
            if radius > res:
                raise Exception("Radius is too large for the resolution")

            #center
            center_x, center_y = np.array(np.random.uniform(0,res*2, size=2),dtype='int32')
            centers_buffer[video_index].append((center_x, center_y))

            # direction: random angle in [0, 2pi]
            dir = np.random.uniform(0, 2*np.pi, 1)
            direction_buffer[video_index].append(dir)

            # speed: random speed in [0.5, 1.5]
            speed = np.random.uniform(1.2, 3.0, 1)
            speed_buffer[video_index].append(speed)
            
            accleration = np.random.uniform(-0.03, 0.03, 1)
            acc_buffer[video_index].append(accleration)

            #predefine an affine transformation matrix
            if opt.affine:
                if shape == 'circle':
                    angle_speed = np.random.uniform(-1/100*np.pi, 1/100*np.pi)
                    scale_x_speed = np.random.uniform(-0.005, 0.005)
                    scale_y_speed = np.random.uniform(-0.005, 0.005)
                    shear_x_speed = np.random.uniform(-0.005, 0.005)
                    shear_y_speed = np.random.uniform(-0.005, 0.005)
                    angle_speed_buffer[video_index].append(angle_speed)
                    scale_x_speed_buffer[video_index].append(scale_x_speed)
                    scale_y_speed_buffer[video_index].append(scale_y_speed)
                    shear_x_speed_buffer[video_index].append(shear_x_speed)
                    shear_y_speed_buffer[video_index].append(shear_y_speed)
                else:
                    angle_speed = np.random.uniform(-1/100*np.pi, 1/100*np.pi)
                    scale_x_speed = np.random.uniform(-0.005, 0.005)
                    scale_y_speed = np.random.uniform(-0.005, 0.005)
                    shear_x_speed = np.random.uniform(-0.005, 0.005)
                    shear_y_speed = np.random.uniform(-0.005, 0.005)
                    angle_speed_buffer[video_index].append(angle_speed)
                    scale_x_speed_buffer[video_index].append(scale_x_speed)
                    scale_y_speed_buffer[video_index].append(scale_y_speed)
                    shear_x_speed_buffer[video_index].append(shear_x_speed)
                    shear_y_speed_buffer[video_index].append(shear_y_speed)

            #get texture file id for each object
            if textures is not None:
                texture_id = np.random.randint(0, len(textures) - 1)
                texture_id_buffer[video_index].append(texture_id)
                

        else:
            color = colors_buffer[video_index][i]
            radius = radius_buffer[video_index][i]
            center_x, center_y = centers_buffer[video_index][i]
            dir = direction_buffer[video_index][i]
            v = speed_buffer[video_index][i]
            if opt.acc:
                acc = acc_buffer[video_index][i]
                v = v + acc * frame
            if opt.static:
                v = 0
            update_x = center_x + v * np.cos(dir) * frame
            update_y = center_y + v * np.sin(dir) * frame
            
            center_x, center_y = np.array((update_x[0], update_y[0]), dtype='int32')
            if textures is not None:
                texture_id = texture_id_buffer[video_index][i]
        if center_x <= -res or center_x >= 2*res or center_y <= -res or center_y >= 2*res:
            continue
        if textures is not None:
            texture_file_path = textures[texture_id]
            add_texture  = cv2_resize(cv2_imread(texture_file_path), (res * 2, res * 2))
        if shape == 'circle':
            rotate_angle = 0
            radius_x = radius
            radius_y = radius
            if opt.affine:
                ori_corners = np.array(((0, radius), (radius, 0), (2 * radius, radius)), dtype='float32')
                c = np.cos(rotate_angle)
                s = np.sin(rotate_angle)
                new_corners = np.array(((center_x - radius_x * c, center_y + radius_x * s), (center_x - radius_y * s, center_y - radius_y * c), (center_x + radius * c, center_y - radius * s)), dtype='float32')
                
                affine_matrix = cv2.getAffineTransform(ori_corners, new_corners)
            img = cv2.ellipse(img, (center_x, center_y), (int(radius_x), int(radius_y)), rotate_angle, 0, 360, color, -1)
            if opt.textured:
                blank = np.zeros((res, res), dtype=np.uint8)
                mask = cv2.ellipse(blank, (center_x, center_y), (int(radius_x), int(radius_y)), rotate_angle, 0, 360, 255, -1)
                if opt.affine:
                    new_texture = cv2.warpAffine(add_texture, affine_matrix, (res, res))
                    masked_texture = cv2.bitwise_and(new_texture, new_texture, mask=mask)
                else:
                    masked_texture = cv2.bitwise_and(add_texture, add_texture, mask=mask)
                img = img + masked_texture

        else:
            if opt.affine:
                affine_c_x, affine_c_y = radius // 2, radius // 2
                angle_speed = angle_speed_buffer[video_index][i]
                rotate_angle = angle_speed * frame
                c, s = np.cos(rotate_angle), np.sin(rotate_angle)
                R = np.array(((c, -s, affine_c_x * (1 - c) + affine_c_y * s), (s, c, affine_c_y * (1 - c) - affine_c_x * s), (0, 0, 1)))
                #scale
                scale_x_speed = scale_x_speed_buffer[video_index][i]
                scale_y_speed = scale_y_speed_buffer[video_index][i]
                scale_x = 1 + scale_x_speed * frame
                scale_y = 1 + scale_y_speed * frame
                S = np.array(((scale_x, 0, affine_c_y * (1 - scale_x)), (0, scale_y, affine_c_y * (1 - scale_y)), (0, 0, 1)))
                #shear
                shear_x_speed = shear_x_speed_buffer[video_index][i]
                shear_y_speed = shear_y_speed_buffer[video_index][i]
                shear_x = shear_x_speed * frame
                shear_y = shear_y_speed * frame
                Sh = np.array(((1, shear_x, -shear_x * affine_c_x), (shear_y, 1, -shear_y * affine_c_y), (0, 0, 1)))
                #affine transformation matrix
                A = R @ S @ Sh
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
                                (- side / 2, + side / 2)),dtype='int32')
                if frame == 0:
                    theta = np.random.uniform(0, 2 * np.pi)
                    theta_buffer[video_index][i] = theta
                else:
                    theta = theta_buffer[video_index][i]
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))
                corners = (R @ corners.transpose()).transpose()
                corners = corners + np.array((radius, radius))
            elif shape == 'rectangle':
                # sample one points in the firrst quadrant, and get the two other symmetric
                if frame == 0:
                    a = np.random.uniform(0, 0.5*np.pi, 1)
                    rec_buffer[video_index][i] = a  
                else:
                    
                    a = rec_buffer[video_index][i]
                corners = np.array(((+ radius * np.cos(a) + radius, + radius * np.sin(a) + radius),
                                    (+ radius * np.cos(a) + radius, - radius * np.sin(a) + radius),
                                    (- radius * np.cos(a) + radius, - radius * np.sin(a) + radius),
                                    (- radius * np.cos(a) + radius, + radius * np.sin(a) + radius)), dtype='int32')[:,:,0]

            else:
                # we sample three or 4 points on a circle of the given radius
                if frame == 0:
                    angles = sorted(np.random.uniform(0, 2*np.pi, 3 if shape == 'triangle' else 4))
                    angle_buffer[video_index][i] = angles                  
                else:
                    angles = angle_buffer[video_index][i]
                corners = []
                for a in angles:
                    corners.append((radius * np.cos(a) + radius, radius * np.sin(a) + radius))
                corners = np.array(corners, dtype='int32')
            if opt.affine:
                ori_corners = corners
                corners_3d = np.concatenate((corners, np.ones((corners.shape[0], 1))), axis=1)
                corners = (A @ corners_3d.transpose()).transpose()[:, :2]

            corners = np.array((center_x, center_y)) + np.array(corners)

            img = cv2.fillPoly(img, np.array(corners, dtype='int32')[None], color=color)
            #masking
            if textures is not None:
                blank = np.zeros((res, res), dtype=np.uint8)
                mask = cv2.fillPoly(blank, np.array(corners, dtype='int32')[None], 255)

                if opt.affine:
                    affine_matrix = cv2.getAffineTransform(np.float32(ori_corners[:3]), np.float32(corners[:3]))
                    new_texture = cv2.warpAffine(add_texture, affine_matrix, (res, res))
                    # new_texture = np.clip(new_texture, 0, 255)
                    masked_texture = cv2.bitwise_and(new_texture, new_texture, mask=mask)
                else:
                    masked_texture = cv2.bitwise_and(add_texture, add_texture, mask=mask)

                img = img + masked_texture

    img = img.transpose((2,0,1))
    
    return np.clip(img, 0, 255)

def generate_dataset(shape_mode, output_path, resolution=96, parallel=True, N_samples=105000, textures=None, frame=0, color_map=None, iteration=0):
    sigma = 3
    to_generate = list(range(N_samples))
    import time
    random.seed(time.time())

    if frame == 0:
        for i in range(N_samples):
            img_folder = output_path + '/' + str(i + iteration * N_samples).zfill(7)
            os.makedirs(img_folder, exist_ok=True)

    #generate imgae for a given frame in video i
    def generate_single_train(i):
        v_index = i + iteration * N_samples
        img_folder = output_path + '/' + str(v_index).zfill(7)
        image_path = img_folder + '/' + str(frame).zfill(7) + '.png'
        error=False
        try:
            if os.path.exists(image_path):
                print("Image {} already exists!".format(i))
        except:
            print("Error in an image {}, will regenerate!".format(i))
            error = True
            pass
        
        im = dead_leaves(resolution, sigma, shape_mode=shape_mode, textures=textures, max_iters=opt.max_it, frame=frame, video_index=v_index)
        if not os.path.exists(image_path) or error:
            # check again for race condition, except if there was a previous error
            cv2_imwrite(im, image_path)
    
    if frame == 0:
        for i in tqdm(to_generate):
            generate_single_train(i)
    else:
        if parallel:      
            p_map(generate_single_train, to_generate)
        else:
            for i in tqdm(to_generate):
                generate_single_train(i)

    


def subdivide_folders(base_folder):
    all_files = sorted(listdir(base_folder, extension='.png'))
    for f in tqdm(all_files):
        id = int(f.split('/')[-1].replace('.png', ''))
        folder_file = base_folder + '/' + str((id // 1000) * 1000).zfill(10)
        os.makedirs(folder_file, exist_ok=True)
        shutil.move(base_folder + '/' + f, folder_file + '/' + f)

def large_scale_options(opt):
    opt.resolution = 256
    # opt.samples = 187
    return opt

def small_scale_options(opt):
    opt.resolution = 128
    opt.samples = 10
    return opt

def get_output_path(opt):
    if len(opt.output_folder) == 0:
        output_path = '../data/{}/dead_leaves-'.format('large_scale_textures')
        if not opt.textured:
            if opt.shape_model == 'mixed':
                output_path += 'mixed/train'
            elif opt.shape_model == 'oriented_square':
                output_path += 'oriented/train'
            elif opt.shape_model == 'square':
                output_path += 'squares/train'
            elif opt.shape_model == 'circle':
                output_path += 'circles/train'
            else:
                raise Exception("Not one of the default modes, so output_path has to be passed as an argument!")
        elif opt.texture_folder.endswith('stat-spectrum_color_wmm/train'):
            # output_path += 'textures/train'
            output_path += 'textures_synthetic/'
            if opt.shape_model == 'mixed':
                output_path += 'mixed/train'
            elif opt.shape_model == 'oriented_square':
                output_path += 'oriented/train'
            elif opt.shape_model == 'square':
                output_path += 'squares/train'
            elif opt.shape_model == 'circle':
                output_path += 'circles/train'
            elif opt.shape_model == 'rectangle':
                output_path += 'rectangle/train'
            elif opt.shape_model == 'triangle':
                output_path += 'triangle/train'
            else:
                raise Exception("Not one of the default modes, so output_path has to be passed as an argument!")
        elif 'imagenet_full_size' in opt.texture_folder:
            output_path += 'textures_imagenet/train'
        elif 'stylegan' in opt.texture_folder:
            output_path += 'textures_stylegan/train'
            # raise Exception("Not one of the default modes, so output_path has to be passed as an argument!")
    else:
        output_path = opt.output_folder
    return output_path

def generate_video(opt, iteration=0): 
    to_generate = list(range(opt.samples))
    def generate_single_video(i):
        index = i + iteration * opt.samples
        data_path = opt.output_folder + '/' + str(index).zfill(7)
        fps = opt.fps          
        size = (opt.resolution, opt.resolution)
        video_path = opt.video_folder + '/' + str(index).zfill(7) + '.mp4' 
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, fps, size)
        
        #only use a subset of the frames
        frames = np.random.randint(100, 200)
        for i in range(frames):      
            image_path = data_path + '/' + str(i).zfill(7) + '.png'
            img = cv2.imread(image_path)
            video.write(img)
        
        video.release()
        cv2.destroyAllWindows()
        #delete the image directory
        # shutil.rmtree(data_path)
    
    if opt.parallel:
        p_map(generate_single_video, to_generate)
    else:
        for i in tqdm(to_generate):
            generate_single_video(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Synthetic Video Generation')

    parser.add_argument('--small-scale', action="store_true", help="use small-scale default parameters")
    parser.add_argument('--large-scale', action="store_true", help="use large-scale default parameters")

    parser.add_argument('--output-folder', default='', type=str, help='Output folder where to dump the datasets')
    parser.add_argument('--output-file-type', type=str, default='png', choices=['jpg', 'png'], help='Filetype to generate')

    parser.add_argument('--resolution', type=int, default=128, help='Resolution to use')
    parser.add_argument('--samples', type=int, default=100, help='N samples to generate')
    parser.add_argument('--samples_it', type=int, default=51, help='rounds of generation')

    parser.add_argument('--textured', action="store_true", help="use textures")
    parser.add_argument('--texture-folder', default='PATH_TO_TEXTURES', help='texture folder to load')

    parser.add_argument('--parallel', type=str2bool, default="False", help='Whether to apply a random rotation to the color channels so that they are correlated')
    parser.add_argument('--shape-model', type=str, default="square", choices=['square', 'oriented_square', 'circle', 'rectangle', 'mixed', 'triangle'], help='What type of shapes to use')

    parser.add_argument('--fps', type=int, default=25, help='FPS of the generated video')
    parser.add_argument('--duration', type=int, default=8, help='Duration of the generated video')
    parser.add_argument('--max_it', type=int, default=500, help='max objects per image')
    parser.add_argument('--worker_id', type=int, default=0, help='worker_id')

    parser.add_argument('--affine', action="store_true", help='Whether to apply affine transform')
    parser.add_argument('--acc', action="store_true", help='Whether to apply acceleration')
    parser.add_argument('--static', action="store_true", help='Whether to use static images')
    
    

    opt = parser.parse_args()

    if opt.large_scale:
        opt = large_scale_options(opt)
    elif opt.small_scale:
        opt = small_scale_options(opt)
    output_folder = get_output_path(opt)
    opt.output_folder = output_folder
    opt.video_folder = output_folder.replace('train', 'video')

    if not os.path.exists(opt.video_folder):
        os.makedirs(opt.video_folder, exist_ok=True)


    
    if opt.textured:
        
        textures_wmm_folders = sorted(listdir(opt.texture_folder, prepend_folder=True))
        texture_files = []
        print("listing textures wmm")
        
        for dir in tqdm(textures_wmm_folders):
            texture_files.extend(listdir(dir, prepend_folder=True))
        print("End listing textures wmm")

        random.shuffle(texture_files)
        
        #alternatively, load part of the textures to memory
        # n_textures_to_load = 30000
        # assert len(texture_files) >= n_textures_to_load
        # texture_files = random.sample(texture_files, n_textures_to_load)
        # loaded_textures = []
        
        # # print(texture_files)
        # # exit()
        # print("Loading textures to memory!")
        # for texture_f in tqdm(texture_files):
        #     loaded_textures.append(cv2_resize(cv2_imread(texture_f), (opt.resolution * 2, opt.resolution * 2)))
        
        # print("Ended textures to memory!")
    else:
        loaded_textures = None
    
    # color_map = get_color_map(len(loaded_textures))
    color_map = None
    frames = opt.fps * opt.duration
    iteration = 0


    #10 chunk per process, each chunk opt.samples of videos
    worker_id = opt.worker_id
    start = worker_id * 10
    end = worker_id * 10 + 10

    print("generate video from {} to {}".format(start, end))
    for it in range(start, end):
        for j in range(frames):
            generate_dataset(opt.shape_model, output_folder, resolution=opt.resolution, textures=texture_files, N_samples=opt.samples, parallel=opt.parallel, frame=j, color_map=color_map, iteration=it)
        generate_video(opt, it)
        print("Iteration {} done!".format(it))

    
