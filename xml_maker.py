import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .box_generator import BoundingBoxGenerator
from .yolo_xml_generator import xml_fill


#ImagePath is directory of directories with first folder as image(countaining .jpeg) and second as masks containing binary masks , like stage1_test_solution
def get_xml_combined_mask(ImagePath, classname, save_image_path , save_annot_path , image_shape = (512 , 512)):
	try:
		os.mkdir(save_image_path)
	except:
		pass
	try:
		os.mkdir(save_annot_path)
	except:
		pass
	image_list = os.listdir(ImagePath)
	for i,elem in enumerate(os.listdir(ImagePath)):
		if elem == '.DS_Store':
			image_list.pop(i)
			break

	for dirs in image_list:
		filler = xml_fill('test_image_folder' + '/' + dirs + '_mask.jpeg', image_shape[0] , image_shape[1])
		maximum = np.zeros(image_shape)
		for img in os.listdir(ImagePath + '/' + dirs + '/' + 'masks'):
			if img == ".DS_Store":
				continue
			else:
				image = plt.imread(ImagePath + '/' + dirs + '/' + 'masks' +'/'+ img)
				image = cv2.resize(image , (512,512))
				#print(image.shape)
				maximum = np.maximum(maximum,image)
				#print(sum(sum(maximum)))
				x, y, w, h = BoundingBoxGenerator(ImagePath + '/' + dirs + '/' + 'masks' +'/'+ img)
				filler.addBox(classname, x, y, w, h)
		_ , thresh = cv2.threshold(maximum ,0.5,255,cv2.THRESH_BINARY)
		print(sum(sum(thresh)))
		im = Image.fromarray(thresh)
		if im.mode != 'RGB':
			im = im.convert('RGB')
			im.save(save_image_path+ '/' + dirs + '_mask.jpeg')
		else:
			im.save(save_image_path + '/' + dirs + '_mask.jpeg')
		filler.save(save_annot_path + '/' + dirs + 'mask.xml')

