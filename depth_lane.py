import argparse
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

cam_port=1


from utils.utils import \
    time_synchronized, select_device, increment_path, \
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, \
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, \
    AverageMeter, \
    LoadImages


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser



device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


model_type = "MiDaS_small" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)


midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def get_depth(frame):



   
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(frame_rgb).to(device)

    
    with torch.no_grad():
        prediction = midas(input_batch)
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map_gray = (depth_map / depth_map.max() * 255).astype('uint8')
    mean_depth = np.mean(depth_map_gray / 255.0)
    
    # Resize depth map to match webcam video size
    # depth_map_resized = cv2.resize(depth_map_gray, (frame.shape[1], frame.shape[0]))
    
    # # Create a horizontal stack of webcam video and depth map
    # output_frame = cv2.hconcat([frame, cv2.cvtColor(depth_map_resized, cv2.COLOR_GRAY2BGR)])
    
    return depth_map_gray,mean_depth


 









def detect():
    # setting and directories
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if source == '0':
        webcam = True
        vid_cap =cv2.VideoCapture(cam_port)
    else:
        webcam = False
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    while True:
        if webcam:
            ret_val, img = vid_cap.read()
            depth_frame,mean_depth = get_depth(img)
            if not ret_val:
                print("Webcam read failed. Exiting...")
                break
            img0 = img.copy()
            img0 = cv2.resize(img0, (1280,720), interpolation=cv2.INTER_LINEAR)
        else:
            try:
                path, img, im0s, vid_cap = next(dataset)
            except StopIteration:
                print("No more images to process. Exiting...")
                break
            img0 = im0s

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img.permute(0,3,1,2))
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # for i, det in enumerate(pred):
        #     if len(det):
        #         for *xyxy, conf, cls in reversed(det):
        #             plot_one_box(xyxy, img0, line_thickness=3)

        show_seg_result(img0, (da_seg_mask,ll_seg_mask), is_demo=True)
        


        # Resize depth map to match webcam video size
        img0=cv2.resize(img0, (img0.shape[1]//2, img0.shape[0]//2))
        depth_map_resized = cv2.resize(depth_frame, (img0.shape[1], img0.shape[0]))
        
        # Create a horizontal stack of webcam video and depth map
        output_frame = cv2.hconcat([img0, cv2.cvtColor(depth_map_resized, cv2.COLOR_GRAY2BGR)])
        mean_depth_text = f"Mean Depth: {mean_depth:.2f}"
        text_size, _ = cv2.getTextSize(mean_depth_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        text_x = (output_frame.shape[1] - text_size[0]) // 2
        cv2.putText(output_frame, mean_depth_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


        # Show the combined frame in a single window
        cv2.imshow("Webcam + Depth Map", output_frame)



      

        # Check for 'q' key to quit
        if cv2.waitKey(1) == ord('q'):
            break
        
            

    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')


        


if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()


    

        
