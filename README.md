# Convert-To-YOLO-XML

Script to Generate Bounding Boxes for Images with Binary Masks and Convert Bounding Box Annotations into XML in VOC format as in http://host.robots.ox.ac.uk/pascal/VOC/ useful for training YOLO algorithm

Download Repo (Assuming Python 3)


    git clone https://github.com/Fenil3510/convert_to_yolo_xml
   

Start Python Session
   
    from convert_to_yolo_xml.xml_maker import get_xml_combined_mask

Directory structure for images and corresponding masks should be of form

    ImagePath
    |__ Image1_id
                |__ image
                        |__ Image_name.jpg / .png
                |___ masks
                        |__ Binary Mask .jpg/.png files
     |__ Image2_id
                |__ image
                        |__ Image_name.jpg / .png
                |___ masks
                        |__ Binary Mask .jpg/.png files
     |__ Image3_id
      .
      .
      .

Now call the function, Default Image Shape is (512,512)
  
    get_xml_combined_mask(ImagePath , Classname(String) , path_to_save_combined_mask , path_to_save_xml , image_shape= (512,512))

Combined Mask and Corresponding XML files will be saved in respective folders.
