import argparse
import os
import glob
import random
import numpy as np
import torch
from networks.model import VRBModel
from networks.traj import TrajAffCVAE
from inference import run_inference
from PIL import Image

def main(args):
    epick_100 = glob.glob('/iris/u/oliviayl/repos/affordance-learning/epic_kitchens/DATASETS/EPIC-KITCHENS-100/folder*/*')
    epick_2018 = glob.glob('/iris/u/oliviayl/repos/affordance-learning/epic_kitchens/DATASETS/EPIC-KITCHENS-2018/frames_rgb_flow/rgb/folder*/*')
    all_epick = epick_100 + epick_2018
    valid_videos = []
    for fp in all_epick:
        video = fp[fp.rfind('/')+1:]
        if os.path.isdir(os.path.join('/iris/u/oliviayl/repos/affordance-learning/epic_kitchens/frame_data/', video)):
            valid_videos.append(fp)
    if len(valid_videos) != len(glob.glob('/iris/u/oliviayl/repos/affordance-learning/epic_kitchens/frame_data/P*')):
        valid_videos_vidnames = [v[v.rfind('/')+1:] for v in valid_videos]
        frame_data_vidnames = [v[v.rfind('/')+1:] for v in frame_data]
        print(set(valid_videos_vidnames) - set(frame_data_vidnames))
    else:
        print('all videos accounted for')

    torch.cuda.manual_seed_all(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    random.seed(args.manual_seed)

    hand_head = TrajAffCVAE(in_dim=2*args.traj_len, hidden_dim=args.hidden_dim,
                        latent_dim=args.hand_latent_dim, condition_dim=args.cond_dim,
                        coord_dim=args.coord_dim, traj_len=args.traj_len)
        
    #resnet output
    if args.resnet_type == 'resnet50': 
        src_in_features = 2048
    else: 
        src_in_features = 512

    net = VRBModel(src_in_features=src_in_features,
                            num_patches=1,
                            hidden_dim=args.hidden_dim, 
                            hand_head=hand_head,
                            encoder_time_embed_type=args.encoder_time_embed_type,
                            num_frames_input=10,
                            resnet_type=args.resnet_type, 
                            embed_dim=args.cond_dim, coord_dim=args.coord_dim,
                            num_heads=args.num_heads, enc_depth=args.enc_depth, 
                            attn_kp=args.attn_kp, attn_kp_fc=args.attn_kp_fc, n_maps=5)

    dt = torch.load(args.model_path, map_location='cpu')
    net.load_state_dict(dt)
    net = net.cpu()
    # for vid_path in valid_videos:
    #     video = vid_path[vid_path.rfind('/')+1:]
    #     if video == 'P02_132': continue
    #     images = glob.glob(vid_path + '/*.jpg')
    #     for img in images:
    #         torch.cuda.empty_cache()
    #         image_pil = Image.open(img).convert("RGB")
    #         image_pil = image_pil.resize((1008, 756))
    #         objs = []
    #         with open(os.path.join(args.visor_objs, video, 'obj_list.txt'), 'r') as fp:
    #             for line in fp:
    #                 line = line[:-1] # remove /n
    #                 if line.find('/') == -1:
    #                     objs.append(line)
    #                 else: 
    #                     line_split = line.split('/')
    #                     objs.extend(line_split)
            im_out = run_inference(net, image_pil) # run_inference(net, image_pil, objs)
            if not os.path.exists(os.path.join(args.save_dir, video)):
                os.makedirs(os.path.join(args.save_dir, video))
            im_out.save(os.path.join(args.save_dir, video, img[img.rfind('/')+1:-4] + '_out.png'))
            print(img)
        print(video)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_heads', type=int, default=8, help='num of heads in transformer')
    parser.add_argument('--enc_depth', type=int, default=6, help='transformer encoder depth')
    parser.add_argument('--hidden_dim', type=int, default=192, help="hidden feature dimension")
    parser.add_argument('--hand_latent_dim', type=int, default=4, help="Latent dimension for trajectory CVAE")
    parser.add_argument('--cond_dim', type=int, default=256, help="downprojection dimension for transformer encoder")
    parser.add_argument('--coord_dim', type=int, default=64, help='Contact coordinate feature dimension')
    parser.add_argument('--resnet_type', type=str, default='resnet18')
    parser.add_argument('--attn_kp', type=int, default=1)
    parser.add_argument('--attn_kp_fc', type=int, default=1)
    parser.add_argument('--traj_len', type=int, default=5)
    parser.add_argument("--encoder_time_embed_type", default="sin",  choices=["sin", "param"], help="transformer encoder time position embedding")
    parser.add_argument("--manual_seed", default=0, type=int, help="manual seed")
    # parser.add_argument('--image', type=str, default='./kitchen.jpeg')
    # parser.add_argument('--video', type=str, default='example')
    parser.add_argument('--model_path', type=str, default='./models/model_checkpoint_1249.pth.tar')
    parser.add_argument('--visor_objs', type=str, default='/iris/u/oliviayl/repos/affordance-learning/epic_kitchens/frame_data')
    parser.add_argument('--save_dir', type=str, default='./results/')
    args = parser.parse_args()
    

    main(args)
    print("All done !")