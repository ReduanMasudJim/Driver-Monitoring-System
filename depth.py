import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt




device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


model_type = "MiDaS_small" 
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.to(device)


midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


cap = cv2.VideoCapture('http://192.168.0.106:4747/video')
while True:
    ret, frame = cap.read()  
    print(frame.shape)


   
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


 

# Convert depth map to grayscale
    depth_map = prediction.cpu().numpy()
    depth_map_gray = (depth_map / depth_map.max() * 255).astype('uint8')
    
    # Resize depth map to match webcam video size
    depth_map_resized = cv2.resize(depth_map_gray, (frame.shape[1], frame.shape[0]))
    
    # Create a horizontal stack of webcam video and depth map
    output_frame = cv2.hconcat([frame, cv2.cvtColor(depth_map_resized, cv2.COLOR_GRAY2BGR)])
    
    # Display the combined video
    cv2.imshow('Webcam + Depth Map', output_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
    

cap.release()
cv2.destroyAllWindows()