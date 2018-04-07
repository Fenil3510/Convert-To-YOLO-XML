import cv2
from box_generator import BoundingBoxGenerator
from yolo_xml_generator import xml_fill

def get_xml(ImagePath, classname, save_path):
	x, y, w, h = BoundingBoxGenerator(ImagePath)
	current_image = cv2.imread(ImagePath , cv2.IMREAD_COLOR)
	filler = xml_fill(ImagePath, current_image.shape[0], current_image.shape[1])
	filler.addBox(classname, x, y, w, h)
	filler.save(save_path)