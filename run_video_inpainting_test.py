import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '..')))

import argparse
import copy
import cv2
import glob
import imageio
import numpy as np
import os
import scipy.ndimage
import torch
import torchvision.transforms.functional as F
from PIL import Image
from skimage.feature import canny

# Custom
import edgeconnect.utils
import PIL.ImageOps
import shutil
import skimage.io
import skimage.util
import warnings
from logging import warning as warn

# RAFT
from RAFT import utils
from RAFT import RAFT

# EdgeConnect
from edgeconnect.networks import EdgeGenerator_

# tool
from tool.get_flowNN import get_flowNN
from tool.get_flowNN_gradient import get_flowNN_gradient
from tool.spatial_inpaint import spatial_inpaint
from tool.frame_inpaint import DeepFillv1

# utils
import utils.region_fill as rf
from utils.Poisson_blend import Poisson_blend
from utils.Poisson_blend_img import Poisson_blend_img
from utils.common_utils import flow_edge


# Custom root warnings.
def _precision_warn(p1, p2, extra=""):
    msg = (
        "Lossy conversion from {} to {}. {}"
        "Convert image to {} prior to saving to suppress this warning."
    )
    warn(msg.format(p1, p2, extra, p2))

def silence_imageio_warning(*args, **kwarge):
    pass


# FGVC functions
def to_tensor(img):
    img = Image.fromarray(img)
    img_t = F.to_tensor(img).float()
    return img_t


def infer(args, EdgeGenerator, device, flow_img_gray, edge, mask):

    # Add a pytorch dataloader
    flow_img_gray_tensor = to_tensor(flow_img_gray)[None, :, :].float().to(device)
    edge_tensor = to_tensor(edge)[None, :, :].float().to(device)
    mask_tensor = torch.from_numpy(mask.astype(np.float64))[None, None, :, :].float().to(device)

    # Complete the edges
    edges_masked = (edge_tensor * (1 - mask_tensor))
    images_masked = (flow_img_gray_tensor * (1 - mask_tensor)) + mask_tensor
    inputs = torch.cat((images_masked, edges_masked, mask_tensor), dim=1)

    with torch.no_grad():
        edges_completed = EdgeGenerator(inputs) # in: [grayscale(1) + edge(1) + mask(1)]
    edges_completed = edges_completed * mask_tensor + edge_tensor * (1 - mask_tensor)
    edge_completed = edges_completed[0, 0].data.cpu().numpy()
    edge_completed[edge_completed < 0.5] = 0
    edge_completed[edge_completed >= 0.5] = 1

    return edge_completed


def gradient_mask(mask):

    gradient_mask = np.logical_or.reduce((mask,
        np.concatenate((mask[1:, :], np.zeros((1, mask.shape[1]), dtype=np.bool)), axis=0),
        np.concatenate((mask[:, 1:], np.zeros((mask.shape[0], 1), dtype=np.bool)), axis=1)))

    return gradient_mask


def create_dir(dir):
    """Creates a directory if not exist.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def initialize_RAFT(args):
    """Initializes the RAFT model.
    """
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to('cuda')
    model.eval()

    return model


def calculate_flow(args, model, video, mode):
    """1. Calculates optical flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    nFrame, _, imgH, imgW = video.shape
    Flow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)

    # If already exist flow, load and return.
    if os.path.isdir(os.path.join(args.outroot, '1_flow', mode + '_flo')):
        for flow_name in sorted(glob.glob(os.path.join(args.outroot, '1_flow', mode + '_flo', '*.flo'))):
            print("Loading {0}".format(flow_name), '\r', end='')
            flow = utils.frame_utils.readFlow(flow_name)
            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)
        return Flow

    create_dir(os.path.join(args.outroot, '1_flow', mode + '_flo'))
    create_dir(os.path.join(args.outroot, '1_flow', mode + '_png'))
    flow_gif = []

    with torch.no_grad():
        for i in range(video.shape[0] - 1):
            print("Calculating {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
            if mode == 'forward':
                # Flow i -> i + 1
                image1 = video[i, None]
                image2 = video[i + 1, None]
            elif mode == 'backward':
                # Flow i + 1 -> i
                image1 = video[i + 1, None]
                image2 = video[i, None]
            else:
                raise NotImplementedError

            _, flow = model(image1, image2, iters=20, test_mode=True)
            flow = flow[0].permute(1, 2, 0).cpu().numpy()
            Flow = np.concatenate((Flow, flow[..., None]), axis=-1)

            # Flow visualization.
            flow_img = utils.flow_viz.flow_to_image(flow)
            flow_img = Image.fromarray(flow_img)

            # Saves the flow and flow_img.
            flow_img.save(os.path.join(args.outroot, '1_flow', mode + '_png', '%05d.png'%i))
            utils.frame_utils.writeFlow(os.path.join(args.outroot, '1_flow', mode + '_flo', '%05d.flo'%i), flow)

            # Convert image to gif
            flow_gif.append(imageio.imread(os.path.join(args.outroot, '1_flow', mode + '_png', '%05d.png' % i)))

        # Save gif.
        imageio.mimsave(os.path.join(args.outroot, '0_process', '1_flow_' + mode + '.gif'), flow_gif, format='gif', fps=20)

    return Flow


def edge_completion(args, EdgeGenerator, corrFlow, flow_mask, mode):
    """2. Calculate flow edge and complete it.
    """   

    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    imgH, imgW, _, nFrame = corrFlow.shape
    Edge = np.empty(((imgH, imgW, 0)), dtype=np.float32)
    
    # If already exist edge, load and return.
    if os.path.isdir(os.path.join(args.outroot, '3_edge_comp', mode + '_npy')):
        for edge_name in sorted(glob.glob(os.path.join(args.outroot, '3_edge_comp', mode + '_npy', '*.npy'))):
            print("Loading {0}".format(edge_name), '\r', end='')
            edge = np.load(edge_name)
            Edge = np.concatenate((Edge, edge[..., None]), axis=-1)
        return Edge

    create_dir(os.path.join(args.outroot, '2_edge_canny', mode + '_png'))
    create_dir(os.path.join(args.outroot, '3_edge_comp', mode + '_npy'))
    create_dir(os.path.join(args.outroot, '3_edge_comp', mode + '_png'))
    canny_gif = []
    edge_comp_gif = []

    if args.merge:
        create_dir(os.path.join(args.outroot, '3_edge_comp', mode + '_merge_png'))
        edge_merge_gif = []

    for i in range(nFrame):
        print("Completing {0} flow edge {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
        flow_mask_img = flow_mask[:, :, i] if mode == 'forward' else flow_mask[:, :, i + 1]
        flow_img_gray = (corrFlow[:, :, 0, i] ** 2 + corrFlow[:, :, 1, i] ** 2) ** 0.5
        flow_img_gray = flow_img_gray / flow_img_gray.max()

        # Complete edge connection
        edge_corr = canny(flow_img_gray, sigma=2, mask=(1 - flow_mask_img).astype(np.bool))
        edge_completed = infer(args, EdgeGenerator, torch.device('cuda:0'), flow_img_gray, edge_corr, flow_mask_img)
        Edge = np.concatenate((Edge, edge_completed[..., None]), axis=-1)
        
        # Save the edge.
        np.save(os.path.join(args.outroot, '3_edge_comp', mode + '_npy', '%05d' % i), edge_completed)
        
        # Extract and Save canny edge.
        img_canny = canny(flow_img_gray, sigma=2, mask=(1- flow_mask_img).astype(np.bool))
        img_canny = skimage.util.img_as_ubyte(img_canny)
        img_canny = cv2.bitwise_not(img_canny)
        skimage.io.imsave(os.path.join(args.outroot, '2_edge_canny', mode + '_png', '%05d.png' % i), img_canny)

        # Extract edge connect completion.
        img_edge = edge_completed
        img_edge = np.array(img_edge)
        img_edge = skimage.util.img_as_ubyte(img_edge)
        img_edge = cv2.bitwise_not(img_edge)
        skimage.io.imsave(os.path.join(args.outroot, '3_edge_comp', mode + '_png', '%05d.png' % i), img_edge)

        # Merge edges with color.
        if args.merge:
            img_canny = Image.open(os.path.join(args.outroot, '2_edge_canny', mode + '_png', '%05d.png' % i)).convert('RGB')
            img_edge_comp = Image.open(os.path.join(args.outroot, '3_edge_comp', mode + '_png', '%05d.png' % i)).convert('RGB')
            
            color_black = (0, 0, 0)
            color_new = (255, 0, 0)

            for x in range(imgW):
                for y in range(imgH):
                    if img_edge_comp.getpixel((x, y)) == color_black:
                        img_edge_comp.putpixel((x, y), color_new)
                
                    if img_canny.getpixel((x, y)) == color_black:
                        img_edge_comp.putpixel((x, y), color_black)

            # Save image and Convert image to gif.
            img_edge_comp.save(os.path.join(args.outroot, '3_edge_comp', mode + '_merge_png', '%05d.png' % i))
            edge_merge_gif.append(imageio.imread(os.path.join(args.outroot, '3_edge_comp', mode + '_merge_png', '%05d.png' % i)))

        # Convert image to gif
        canny_gif.append(imageio.imread(os.path.join(args.outroot, '2_edge_canny', mode + '_png', '%05d.png' % i)))
        edge_comp_gif.append(imageio.imread(os.path.join(args.outroot, '3_edge_comp', mode + '_png', '%05d.png' % i)))

    # Save gif.
    imageio.mimsave(os.path.join(args.outroot, '0_process', '2_canny_' + mode + '.gif'), canny_gif, format='gif', fps=20)
    imageio.mimsave(os.path.join(args.outroot, '0_process', '3_edge_comp_' + mode + '.gif'), edge_comp_gif, format='gif', fps=20)
    
    if args.merge:
        imageio.mimsave(os.path.join(args.outroot, '0_process', '3_edge_comp_' + mode + '_merge.gif'), edge_merge_gif, format='gif', fps=20)
        
    return Edge


def complete_flow(args, corrFlow, flow_mask, mode, edge=None):
    """3. Completes flow.
    """
    if mode not in ['forward', 'backward']:
        raise NotImplementedError

    imgH, imgW, _, nFrame = corrFlow.shape

    # If already exist flow_comp, load and return.
    if os.path.isdir(os.path.join(args.outroot, '4_flow_comp', mode + '_flo')):
        compFlow = np.empty(((imgH, imgW, 2, 0)), dtype=np.float32)
        for flow_name in sorted(glob.glob(os.path.join(args.outroot, '4_flow_comp', mode + '_flo', '*.flo'))):
            print("Loading {0}".format(flow_name), '\r', end='')
            flow = utils.frame_utils.readFlow(flow_name)
            compFlow = np.concatenate((compFlow, flow[..., None]), axis=-1)
        return compFlow

    create_dir(os.path.join(args.outroot, '4_flow_comp', mode + '_flo'))
    create_dir(os.path.join(args.outroot, '4_flow_comp', mode + '_png'))
    flow_comp_gif = []

    compFlow = np.zeros(((imgH, imgW, 2, nFrame)), dtype=np.float32)

    for i in range(nFrame):
        print("Completing {0} flow {1:2d} <---> {2:2d}".format(mode, i, i + 1), '\r', end='')
        flow = corrFlow[:, :, :, i]
        flow_mask_img = flow_mask[:, :, i] if mode == 'forward' else flow_mask[:, :, i + 1]
        flow_mask_gradient_img = gradient_mask(flow_mask_img)

        if edge is not None:
            # imgH x (imgW - 1 + 1) x 2
            gradient_x = np.concatenate((np.diff(flow, axis=1), np.zeros((imgH, 1, 2), dtype=np.float32)), axis=1)
            # (imgH - 1 + 1) x imgW x 2
            gradient_y = np.concatenate((np.diff(flow, axis=0), np.zeros((1, imgW, 2), dtype=np.float32)), axis=0)

            # concatenate gradient_x and gradient_y
            gradient = np.concatenate((gradient_x, gradient_y), axis=2)

            # We can trust the gradient outside of flow_mask_gradient_img
            # We assume the gradient within flow_mask_gradient_img is 0.
            gradient[flow_mask_gradient_img, :] = 0

            # Complete the flow
            imgSrc_gy = gradient[:, :, 2 : 4]
            imgSrc_gy = imgSrc_gy[0 : imgH - 1, :, :]
            imgSrc_gx = gradient[:, :, 0 : 2]
            imgSrc_gx = imgSrc_gx[:, 0 : imgW - 1, :]
            compFlow[:, :, :, i] = Poisson_blend(flow, imgSrc_gx, imgSrc_gy, flow_mask_img, edge[:, :, i])

        else:
            flow[:, :, 0] = rf.regionfill(flow[:, :, 0], flow_mask_img)
            flow[:, :, 1] = rf.regionfill(flow[:, :, 1], flow_mask_img)
            compFlow[:, :, :, i] = flow

        # Flow visualization.
        flow_img = utils.flow_viz.flow_to_image(compFlow[:, :, :, i])
        flow_img = Image.fromarray(flow_img)

        # Saves the flow and flow_img.
        flow_img.save(os.path.join(args.outroot, '4_flow_comp', mode + '_png', '%05d.png'%i))
        utils.frame_utils.writeFlow(os.path.join(args.outroot, '4_flow_comp', mode + '_flo', '%05d.flo'%i), compFlow[:, :, :, i])

        # Convert image to gif.
        flow_comp_gif.append(imageio.imread(os.path.join(args.outroot, '4_flow_comp', mode + '_png', '%05d.png' % i)))

    # Save gif.
    imageio.mimsave(os.path.join(args.outroot, '0_process', '4_flow_comp_' + mode + '.gif'), flow_comp_gif, format='gif', fps=20)

    return compFlow


def video_completion(args):

    # Flow model.
    RAFT_model = initialize_RAFT(args)

    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    # Obtains imgH, imgW and nFrame.
    imgH, imgW = np.array(Image.open(filename_list[0])).shape[:2]
    nFrame = len(filename_list)
    print('Image Size : {0} x {1} x {2} frames'.format(imgH, imgW, nFrame))

    # Loads video.
    video = []
    for filename in sorted(filename_list):
        video.append(torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)[..., :3]).permute(2, 0, 1).float())

    video = torch.stack(video, dim=0)
    video = video.to('cuda')

    # Calcutes the corrupted flow.
    corrFlowF = calculate_flow(args, RAFT_model, video, 'forward')
    corrFlowB = calculate_flow(args, RAFT_model, video, 'backward')
    print('\nFinish flow prediction.')

    # Makes sure video is in BGR (opencv) format.
    video = video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.

    '''Object removal without seamless
    '''
    # Loads masks.
    filename_list = glob.glob(os.path.join(args.path_mask, '*.png')) + \
                    glob.glob(os.path.join(args.path_mask, '*.jpg'))

    mask = []
    flow_mask = []
    for filename in sorted(filename_list):
        mask_img = np.array(Image.open(filename).convert('L'))
        mask.append(mask_img)

        # Dilate 15 pixel so that all known pixel is trustworthy
        flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
        # Close the small holes inside the foreground objects
        flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(np.bool)
        flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.bool)
        flow_mask.append(flow_mask_img)

    # mask indicating the missing region in the video.
    mask = np.stack(mask, -1).astype(np.bool)
    flow_mask = np.stack(flow_mask, -1).astype(np.bool)

    if args.edge_guide:
        # Edge completion model.
        EdgeGenerator = EdgeGenerator_()
        EdgeComp_ckpt = torch.load(args.edge_completion_model)
        EdgeGenerator.load_state_dict(EdgeComp_ckpt['generator'])
        EdgeGenerator.to(torch.device('cuda:0'))
        EdgeGenerator.eval()

        # Edge completion.
        FlowF_edge = edge_completion(args, EdgeGenerator, corrFlowF, flow_mask, 'forward')
        FlowB_edge = edge_completion(args, EdgeGenerator, corrFlowB, flow_mask, 'backward')
        print('\nFinish edge completion.')
    else:
        FlowF_edge, FlowB_edge = None, None

    # Completes the flow.
    videoFlowF = complete_flow(args, corrFlowF, flow_mask, 'forward', FlowF_edge)
    videoFlowB = complete_flow(args, corrFlowB, flow_mask, 'backward', FlowB_edge)
    print('\nFinish flow completion.')
    
    #return
    iter = 0
    mask_tofill = mask
    video_comp = video

    # Image inpainting model.
    deepfill = DeepFillv1(pretrained_model=args.deepfill_model, image_shape=[imgH, imgW])

    # We iteratively complete the video.
    while(np.sum(mask_tofill) > 0):
        create_dir(os.path.join(args.outroot, '5_frame_comp_' + str(iter)))

        # Color propagation.
        video_comp, mask_tofill, _ = get_flowNN(args,
                                      video_comp,
                                      mask_tofill,
                                      videoFlowF,
                                      videoFlowB,
                                      None,
                                      None)

        for i in range(nFrame):
            mask_tofill[:, :, i] = scipy.ndimage.binary_dilation(mask_tofill[:, :, i], iterations=2)
            img = video_comp[:, :, :, i] * 255
            # Green indicates the regions that are not filled yet.
            img[mask_tofill[:, :, i]] = [0, 255, 0]
            cv2.imwrite(os.path.join(args.outroot, '5_frame_comp_' + str(iter), '%05d.png'%i), img)

        # video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
        # imageio.mimwrite(os.path.join(args.outroot, 'frame_comp_' + str(iter), 'intermediate_{0}.mp4'.format(str(iter))), video_comp_, fps=12, quality=8, macro_block_size=1)
        # imageio.mimsave(os.path.join(args.outroot, 'frame_comp_' + str(iter), 'intermediate_{0}.gif'.format(str(iter))), video_comp_, format='gif', fps=12)
        mask_tofill, video_comp = spatial_inpaint(deepfill, mask_tofill, video_comp)
        iter += 1

    create_dir(os.path.join(args.outroot, '5_frame_comp_' + 'final'))
    video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
    for i in range(nFrame):
        img = video_comp[:, :, :, i] * 255
        cv2.imwrite(os.path.join(args.outroot, '5_frame_comp_' + 'final', '%05d.png'%i), img)
        imageio.mimwrite(os.path.join(args.outroot, '5_frame_comp_' + 'final', 'final.mp4'), video_comp_, fps=20, quality=8, macro_block_size=1)
        imageio.mimsave(os.path.join(args.outroot, '0_process', '5_frame_comp_final.gif'), video_comp_, format='gif', fps=20)


def video_completion_seamless(args):

    # Flow model.
    RAFT_model = initialize_RAFT(args)

    # Loads frames.
    filename_list = glob.glob(os.path.join(args.path, '*.png')) + \
                    glob.glob(os.path.join(args.path, '*.jpg'))

    # Obtains imgH, imgW and nFrame.
    imgH, imgW = np.array(Image.open(filename_list[0])).shape[:2]
    nFrame = len(filename_list)

    # Loads video.
    video = []
    for filename in sorted(filename_list):
        video.append(torch.from_numpy(np.array(Image.open(filename)).astype(np.uint8)[..., :3]).permute(2, 0, 1).float())

    video = torch.stack(video, dim=0)
    video = video.to('cuda')

    # Calcutes the corrupted flow.
    corrFlowF = calculate_flow(args, RAFT_model, video, 'forward')
    corrFlowB = calculate_flow(args, RAFT_model, video, 'backward')
    print('\nFinish flow prediction.')

    # Makes sure video is in BGR (opencv) format.
    video = video.permute(2, 3, 1, 0).cpu().numpy()[:, :, ::-1, :] / 255.

    '''Object removal with seamless
    '''
    # Loads masks.
    filename_list = glob.glob(os.path.join(args.path_mask, '*.png')) + \
                    glob.glob(os.path.join(args.path_mask, '*.jpg'))

    mask = []
    mask_dilated = []
    flow_mask = []
    for filename in sorted(filename_list):
        mask_img = np.array(Image.open(filename).convert('L'))

        # Dilate 15 pixel so that all known pixel is trustworthy
        flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=15)
        # Close the small holes inside the foreground objects
        flow_mask_img = cv2.morphologyEx(flow_mask_img.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(np.bool)
        flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.bool)
        flow_mask.append(flow_mask_img)

        mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=5)
        mask_img = scipy.ndimage.binary_fill_holes(mask_img).astype(np.bool)
        mask.append(mask_img)
        mask_dilated.append(gradient_mask(mask_img))

    # mask indicating the missing region in the video.
    mask = np.stack(mask, -1).astype(np.bool)
    mask_dilated = np.stack(mask_dilated, -1).astype(np.bool)
    flow_mask = np.stack(flow_mask, -1).astype(np.bool)

    if args.edge_guide:
        # Edge completion model.
        EdgeGenerator = EdgeGenerator_()
        EdgeComp_ckpt = torch.load(args.edge_completion_model)
        EdgeGenerator.load_state_dict(EdgeComp_ckpt['generator'])
        EdgeGenerator.to(torch.device('cuda:0'))
        EdgeGenerator.eval()

        # Edge completion.
        FlowF_edge = edge_completion(args, EdgeGenerator, corrFlowF, flow_mask, 'forward')
        FlowB_edge = edge_completion(args, EdgeGenerator, corrFlowB, flow_mask, 'backward')
        print('\nFinish edge completion.')
    else:
        FlowF_edge, FlowB_edge = None, None

    # Completes the flow.
    videoFlowF = complete_flow(args, corrFlowF, flow_mask, 'forward', FlowF_edge)
    videoFlowB = complete_flow(args, corrFlowB, flow_mask, 'backward', FlowB_edge)
    print('\nFinish flow completion.')

    # Prepare gradients
    gradient_x = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)
    gradient_y = np.empty(((imgH, imgW, 3, 0)), dtype=np.float32)
    
    # If already exist gradient, load and return
    if os.path.isdir(os.path.join(args.outroot, '5_gradient')):
        for grad_x_name in sorted(glob.glob(os.path.join(args.outroot, '5_gradient', 'x_npy', '*.npy'))):
            print("Loading {0}".format(grad_x_name), '\r', end='')
            indFrame = int(grad_x_name.split('.npy')[0][-5:])
            grad_x = np.load(grad_x_name)
            gradient_x = np.concatenate((gradient_x, grad_x.reshape(imgH, imgW, 3, 1)), axis=-1)
            gradient_x[mask_dilated[:, :, indFrame], :, indFrame] = 0
        
        for grad_y_name in sorted(glob.glob(os.path.join(args.outroot, '5_gradient', 'y_npy', '*.npy'))):
            print("Loading {0}".format(grad_y_name), '\r', end='')
            indFrame = int(grad_y_name.split('.npy')[0][-5:])
            grad_y = np.load(grad_y_name)
            gradient_y = np.concatenate((gradient_y, grad_y.reshape(imgH, imgW, 3, 1)), axis=-1)
            gradient_y[mask_dilated[:, :, indFrame], :, indFrame] = 0
    
    else:
        create_dir(os.path.join(args.outroot, '5_gradient', 'x_npy'))
        create_dir(os.path.join(args.outroot, '5_gradient', 'y_npy'))
        create_dir(os.path.join(args.outroot, '5_gradient', 'x_png'))
        create_dir(os.path.join(args.outroot, '5_gradient', 'y_png'))
        grad_x_gif = []
        grad_y_gif = []

        for indFrame in range(nFrame):
            print("Gradient frame {0:2d}".format(indFrame), '\r', end='')
            img = video[:, :, :, indFrame]
            img[mask[:, :, indFrame], :] = 0
            img = cv2.inpaint((img * 255).astype(np.uint8), mask[:, :, indFrame].astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32)  / 255.

            gradient_x_ = np.concatenate((np.diff(img, axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
            gradient_y_ = np.concatenate((np.diff(img, axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)
            gradient_x = np.concatenate((gradient_x, gradient_x_.reshape(imgH, imgW, 3, 1)), axis=-1)
            gradient_y = np.concatenate((gradient_y, gradient_y_.reshape(imgH, imgW, 3, 1)), axis=-1)
        
            gradient_x[mask_dilated[:, :, indFrame], :, indFrame] = 0
            gradient_y[mask_dilated[:, :, indFrame], :, indFrame] = 0
            
            # Save the gradient
            np.save(os.path.join(args.outroot, '5_gradient', 'x_npy', '%05d' % indFrame), gradient_x_)
            np.save(os.path.join(args.outroot, '5_gradient', 'y_npy', '%05d' % indFrame), gradient_y_)
            
            # Extract and Save gradient image
            grad_x = gradient_x[:, :, 0, indFrame]
            grad_y = gradient_y[:, :, 0, indFrame]
            skimage.io.imsave(os.path.join(args.outroot, '5_gradient', 'x_png', '%05d.png' % indFrame), grad_x)
            skimage.io.imsave(os.path.join(args.outroot, '5_gradient', 'y_png', '%05d.png' % indFrame), grad_y)
            
            # print("grad_x shape: {}, dimension: {}".format(grad_x.shape, grad_x.ndim))
            # print("grad_y shape: {}, dimension: {}".format(grad_y.shape, grad_y.ndim))
            
            # Conveert image to gif.
            grad_x_gif.append(imageio.imread(os.path.join(args.outroot, '5_gradient', 'x_png', '%05d.png' % indFrame)))
            grad_y_gif.append(imageio.imread(os.path.join(args.outroot, '5_gradient', 'y_png', '%05d.png' % indFrame)))
            
        # Save gif.
        imageio.mimsave(os.path.join(args.outroot, '0_process', '5_gradient_' + 'x.gif'), grad_x_gif, format='gif', fps=20)
        imageio.mimsave(os.path.join(args.outroot, '0_process', '5_gradient_' + 'y.gif'), grad_y_gif, format='gif', fps=20)

    print('\nFinish gradient frame creation.')

    iter = 0
    mask_tofill = mask
    gradient_x_filled = gradient_x # corrupted gradient_x, mask_gradient indicates the missing gradient region
    gradient_y_filled = gradient_y # corrupted gradient_y, mask_gradient indicates the missing gradient region
    mask_gradient = mask_dilated
    video_comp = video

    # Image inpainting model.
    deepfill = DeepFillv1(pretrained_model=args.deepfill_model, image_shape=[imgH, imgW])

    
    # We iteratively complete the video.
    while(np.sum(mask) > 0):
        create_dir(os.path.join(args.outroot, '6_frame_seamless_comp_' + str(iter)))

        # Gradient propagation.
        
        gradient_x_filled, gradient_y_filled, mask_gradient = \
            get_flowNN_gradient(args,
                                gradient_x_filled,
                                gradient_y_filled,
                                mask,
                                mask_gradient,
                                videoFlowF,
                                videoFlowB,
                                None,
                                None)
        
        create_dir(os.path.join(args.outroot, '6_gradient', 'x_png'))
        create_dir(os.path.join(args.outroot, '6_gradient', 'y_png'))
            
        for indFrame in range(nFrame):
            grad_x_filled = gradient_x_filled[:, :, 0, indFrame]
            grad_y_filled = gradient_y_filled[:, :, 0, indFrame]
            skimage.io.imsave(os.path.join(args.outroot, '6_gradient', 'x_png', '%05d.png' % indFrame), grad_x_filled)
            skimage.io.imsave(os.path.join(args.outroot, '6_gradient', 'y_png', '%05d.png' % indFrame), grad_y_filled)


        # if there exist holes in mask, Poisson blending will fail. So I did this trick. I sacrifice some value. Another solution is to modify Poisson blending.
        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = scipy.ndimage.binary_fill_holes(mask_gradient[:, :, indFrame]).astype(np.bool)

        # After one gradient propagation iteration
        # gradient --> RGB
        for indFrame in range(nFrame):
            print("Poisson blending frame {0:3d}".format(indFrame))

            if mask[:, :, indFrame].sum() > 0:
                '''    
                try:
                    frameBlend, UnfilledMask = Poisson_blend_img(video_comp[:, :, :, indFrame], gradient_x_filled[:, 0 : imgW - 1, :, indFrame], gradient_y_filled[0 : imgH - 1, :, :, indFrame], mask[:, :, indFrame], mask_gradient[:, :, indFrame])
                    UnfilledMask = scipy.ndimage.binary_fill_holes(UnfilledMask).astype(np.bool)
                except:
                    frameBlend, UnfilledMask = video_comp[:, :, :, indFrame], mask[:, :, indFrame]
                '''
                frameBlend, UnfilledMask = video_comp[:, :, :, indFrame], mask[:, :, indFrame]

                frameBlend = np.clip(frameBlend, 0, 1.0)
                tmp = cv2.inpaint((frameBlend * 255).astype(np.uint8), UnfilledMask.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32) / 255.
                frameBlend[UnfilledMask, :] = tmp[UnfilledMask, :]

                video_comp[:, :, :, indFrame] = frameBlend
                mask[:, :, indFrame] = UnfilledMask

                frameBlend_ = copy.deepcopy(frameBlend)
                # Green indicates the regions that are not filled yet.
                frameBlend_[mask[:, :, indFrame], :] = [0, 1., 0]
            else:
                frameBlend_ = video_comp[:, :, :, indFrame]

            cv2.imwrite(os.path.join(args.outroot, '6_frame_seamless_comp_' + str(iter), '%05d.png'%indFrame), frameBlend_ * 255.)

        video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
        # imageio.mimwrite(os.path.join(args.outroot, '6_frame_seamless_comp_' + str(iter), 'intermediate_{0}.mp4'.format(str(iter))), video_comp_, fps=20, quality=8, macro_block_size=1)
        imageio.mimsave(os.path.join(args.outroot, '6_frame_seamless_comp_' + str(iter), 'intermediate_{0}.gif'.format(str(iter))), video_comp_, format='gif', fps=20)
        return
        mask, video_comp = spatial_inpaint(deepfill, mask, video_comp)
        iter += 1

        # Re-calculate gradient_x/y_filled and mask_gradient
        for indFrame in range(nFrame):
            mask_gradient[:, :, indFrame] = gradient_mask(mask[:, :, indFrame])

            gradient_x_filled[:, :, :, indFrame] = np.concatenate((np.diff(video_comp[:, :, :, indFrame], axis=1), np.zeros((imgH, 1, 3), dtype=np.float32)), axis=1)
            gradient_y_filled[:, :, :, indFrame] = np.concatenate((np.diff(video_comp[:, :, :, indFrame], axis=0), np.zeros((1, imgW, 3), dtype=np.float32)), axis=0)

            gradient_x_filled[mask_gradient[:, :, indFrame], :, indFrame] = 0
            gradient_y_filled[mask_gradient[:, :, indFrame], :, indFrame] = 0

    create_dir(os.path.join(args.outroot, '6_frame_seamless_comp_' + 'final'))
    video_comp_ = (video_comp * 255).astype(np.uint8).transpose(3, 0, 1, 2)[:, :, :, ::-1]
    for i in range(nFrame):
        img = video_comp[:, :, :, i] * 255
        cv2.imwrite(os.path.join(args.outroot, '6_frame_seamless_comp_' + 'final', '%05d.png' % i), img)
        imageio.mimwrite(os.path.join(args.outroot, '6_frame_seamless_comp_' + 'final', 'final.mp4'), video_comp_, fps=20, quality=8, macro_block_size=1)
        imageio.mimsave(os.path.join(args.outroot, '0_process', '6_frame_seamless_comp_final.gif'), video_comp_, format='gif', fps=20)


def args_list(args):
    print("\n")
    print("================================================================")
    print("==                            TEST                            ==")
    print("================================================================")
    print("\n")

    args_dict = vars(args)
    for key in args_dict:
        val = args_dict[key]
        print("%s : %s" % (key, val))

    print("")


def main(args):
    assert args.mode in ('object_removal', 'video_extrapolation'), (
        "Accepted modes: 'object_removal', 'video_extrapolation', but input is %s"
    ) % args.mode

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    imageio.core.util._precision_warn = silence_imageio_warning

    args.edge_guide = True
    args_list(args)

    if args.clean:
        shutil.rmtree(os.path.join(args.outroot))

    if args.run:
        create_dir(os.path.join(args.outroot, '0_process'))

        if args.seamless:
            # args.outroot = 'D:/_data/tennis_result_seamless'
            video_completion_seamless(args)
        else:
            video_completion(args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # video completion
    parser.add_argument('--mode', default='object_removal', help="modes: object_removal / video_extrapolation")
    parser.add_argument('--path', default='D:/_data/tennis', help="dataset for evaluation")
    parser.add_argument('--path_mask', default='D:/_data/square_mask', help="mask for object removal")
    parser.add_argument('--outroot', default='D:/_data/tennis_result', help="output directory")

    # options
    parser.add_argument('--seamless', action='store_true', help='Whether operate in the gradient domain')
    parser.add_argument('--edge_guide', action='store_true', help='Whether use edge as guidance to complete flow')
    parser.add_argument('--Nonlocal', dest='Nonlocal', default=False, type=bool)
    parser.add_argument('--alpha', dest='alpha', default=0.1, type=float)
    parser.add_argument('--consistencyThres', dest='consistencyThres', default=np.inf, type=float, help='flow consistency error threshold')

    # RAFT
    parser.add_argument('--model', default='./weight/raft-things.pth', help="restore checkpoint")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

    # Edge completion
    parser.add_argument('--edge_completion_model', default='./weight/edge_completion.pth', help="restore checkpoint")

    # Deepfill
    parser.add_argument('--deepfill_model', default='./weight/imagenet_deepfill.pth', help="restore checkpoint")

    # custom
    parser.add_argument('--run', action='store_true', help='run video completion')
    parser.add_argument('--merge', action='store_true', help='merge image canny edge and completed edge')
    parser.add_argument('--clean', action='store_true', help='clear result directory')

    args = parser.parse_args()
    
    main(args)
