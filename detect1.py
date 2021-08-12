"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
	$ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
	apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

class Detect:
	def __init__(self,matrix):
		self.matrix=matrix
		self.average = [0,0,0,0]

	@torch.no_grad()
	def run(self,
			weights='C:\\Users\\T17932\\Desktop\\CVFlask\\ComputerVisionFlask-main1\\best.pt',  # model.pt path(s)
			source='0',  # file/dir/URL/glob, 0 for webcam
			imgsz=416,  # inference size (pixels)
			conf_thres=0.25,  # confidence threshold
			iou_thres=0.45,  # NMS IOU threshold
			max_det=1000,  # maximum detections per image
			device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
			view_img=False,  # show results
			classes=None,  # filter by class: --class 0, or --class 0 2 3
			agnostic_nms=False,  # class-agnostic NMS
			augment=False,  # augmented inference
			visualize=False,  # visualize features
			update=False,  # update all models
			exist_ok=False,  # existing project/name ok, do not increment
			line_thickness=1,  # bounding box thickness (pixels)
			hide_labels=False,  # hide labels
			hide_conf=True,  # hide confidences
			half=False,  # use FP16 half-precision inference
			):

		source = '0'
		webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
		flag = False
		# Initialize
		set_logging()
		device = select_device(device)

		# Load model
		w = weights #if isinstance(weights, list) else weights
		classify, pt = False, True
		
		stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
		if pt:
			model = attempt_load(weights, map_location=device)  # load FP32 model
			stride = int(model.stride.max())  # model stride
			names = model.module.names if hasattr(model, 'module') else model.names  # get class names
			if classify:  # second-stage classifier
				modelc = load_classifier(name='resnet50', n=2)  # initialize
				modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

		imgsz = check_img_size(imgsz, s=stride)  # check image size

		# Dataloader
		if webcam:
			view_img = check_imshow()
			#view_img = False
			cudnn.benchmark = True  # set True to speed up constant image size inference
			dataset = LoadStreams(source, img_size=imgsz, stride=stride)
			bs = len(dataset)  # batch_size
		vid_path, vid_writer = [None] * bs, [None] * bs

		# Run inference
		if pt and device.type != 'cpu':
			model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
		t0 = time.time()
		for path, img, im0s, vid_cap in dataset:
			if pt:
				img = torch.from_numpy(img).to(device)
				img = img.half() if half else img.float()  # uint8 to fp16/32
			img /= 255.0  # 0 - 255 to 0.0 - 1.0
			if len(img.shape) == 3:
				img = img[None]  # expand for batch dim

			# Inference
			t1 = time_sync()
			if pt:
				pred = model(img, augment=augment, visualize=visualize)[0]

			# NMS
			pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
			t2 = time_sync()

			# Second-stage classifier (optional)
			if classify:
				pred = apply_classifier(pred, modelc, img, im0s)

			my_results = {}
			

			# Process predictions
			for i, det in enumerate(pred):  # detections per image
				if webcam:  # batch_size >= 1
					p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count      
				s += '%gx%g ' % img.shape[2:]  # print string
				gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

				if len(det):
					# Rescale boxes from img_size to im0 size
					det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

					# Print results
					# for c in det[:, -1].unique():
					# 	n = (det[:, -1] == c).sum()  # detections per class
					# 	s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
					   
					# Write results
					for *xyxy, conf, cls in reversed(det):
						c = int(cls)  # integer class
						my_results[int(xyxy[0].cpu().detach().numpy())] = names[c]

						if view_img: 
							c = int(cls)  # integer class
							label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
							plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
					
					#if we found 4 numbers
					if(len(my_results)==self.matrix.shape[1]):
						#get numbers from left to right	
						res = [int(my_results[i]) for i in sorted( my_results.keys() )]
						#put to the matrix
						print("res", res)
						if not flag: #don't have a big number of frames
							if(self.matrix.shape[0] > 1000):
								flag = True
							self.matrix = np.append(self.matrix, [res], axis=0)
						else: #we have enough elements
							#insert at beginning with delete
							self.matrix = np.insert(self.matrix[0:-1], 0, [res], axis=0)
						print("matrix")
						print(*self.matrix)

						self.average = np.round(np.mean(self.matrix,axis=0)).astype(int)
						print("AVERAGE!!", *self.average)



				# Print time (inference + NMS)
				print(f'{s}Done. ({t2 - t1:.3f}s)')

				# Stream results
				if view_img:
					# cv2.imshow(str(p), im0)
					# cv2.waitKey(1)  # 1 millisecond
					ret, buffer = cv2.imencode('.jpg', im0)
					frame = buffer.tobytes()
					yield (b'--frame\r\n'
					   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


	# if update:
	# 	strip_optimizer(weights)  # update model (to fix SourceChangeWarning)

	# print(f'Done. ({time.time() - t0:.3f}s)')


# def parse_opt():
# 	parser = argparse.ArgumentParser()
# 	parser.add_argument('--weights', nargs='+', type=str, default='./best.pt', help='model.pt path(s)')
# 	parser.add_argument('--source', type=str, default='data/images', help='file/dir/URL/glob, 0 for webcam')
# 	parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
# 	parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
# 	parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
# 	parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
# 	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# 	parser.add_argument('--view-img', action='store_true', help='show results')
# 	parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
# 	parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
# 	parser.add_argument('--augment', action='store_true', help='augmented inference')
# 	parser.add_argument('--visualize', action='store_true', help='visualize features')
# 	parser.add_argument('--update', action='store_true', help='update all models')
# 	parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
# 	parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
# 	parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
# 	parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
# 	opt = parser.parse_args()
# 	return opt


# def main(opt):
# 	print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
# 	check_requirements(exclude=('tensorboard', 'thop'))
# 	run(**vars(opt))


# if __name__ == "__main__":
# 	opt = parse_opt()
# 	main(opt)
