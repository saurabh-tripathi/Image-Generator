from matplotlib.pyplot import imshow
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import colorsys
from IPython.display import display
import random
import os
import xml.etree.ElementTree as ET
random.seed(12)


xml_template = r'<annotation verified="yes">' \
               '<folder>Annotation</folder>' \
               '<filename>{fileid}</filename>' \
               '<path>C:\\Users\mhen3\Desktop\\103_output\JPEGImages\{filename}</path>' \
               '<source>' \
               '<database>Unknown</database>' \
               '</source>' \
               '<size>' \
               '<width>{width}</width>' \
               '<height>{height}</height>' \
               '<depth>{depth}</depth>' \
               '</size>' \
               '<segmented>0</segmented>' \
               '<object>' \
               '<name>odometer</name>' \
               '<pose>Unspecified</pose>' \
               '<bndbox>' \
               '<xmin>{x_odo_min}</xmin>' \
               '<ymin>{y_odo_min}</ymin>' \
               '<xmax>{x_odo_max}</xmax>' \
               '<ymax>{y_odo_max}</ymax>' \
               '</bndbox>' \
               '<text>{odo_val}</text>' \
               '</object>' \
               '<object>' \
               '<name>t_odometer</name>' \
               '<pose>Unspecified</pose>' \
               '<bndbox>' \
               '<xmin>{x_t_odo_min}</xmin>' \
               '<ymin>{y_t_odo_min}</ymin>' \
               '<xmax>{x_t_odo_max}</xmax>' \
               '<ymax>{y_t_odo_max}</ymax>' \
               '</bndbox>' \
               '<text>{t_odo_val}</text>' \
               '</object>' \
               '</annotation>'



def change_hue(im):
    width, height = im.size
    ld = im.load()
    hue_shift = random.choice([15.0,25.0, 90.])
    for y in range(height):
        for x in range(width):
            r,g,b = ld[x,y]
            h,s,v = colorsys.rgb_to_hsv(r/255., g/255., b/255.)
            h = (h + -hue_shift/360.0) % 1.0
            r,g,b = colorsys.hsv_to_rgb(h, s, v)
            ld[x,y] = (int(r * 255.9999), int(g * 255.9999), int(b * 255.9999))


def create_dummy_file(filename):
    img = cv2.imread(filename)
    # Create the basic black image
    mask = np.zeros(img.shape, dtype = "uint8")
    # Draw a white, filled rectangle on the mask image
    mask = cv2.rectangle(mask, (350, 330), (450, 380), (255, 255, 255), -1)
    # convert this 3-channel to 1-channel
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    dst = cv2.inpaint(img,gray_mask,3,cv2.INPAINT_TELEA)
    cv2.imwrite('dummy.jpg',dst)
    return 'dummy.jpg'


def add_annotations(annotation_dict):
    pass


def create_annotation():
    list_files = os.listdir('Annotations')
    add_annotations = {}
    for file in list_files:
        tree = ET.parse(os.path.join('Annotations', file))
        width = tree.find('size').find('width').text
        height = tree.find('size').find('height').text
        depth = tree.find('size').find('depth').text
        objs = tree.findall('object')
        num_objs = len(objs)
        image_file = os.path.join('JPEGImages', file.split('.')[0]+'.jpg')
        if num_objs == 2 and os.path.exists(image_file):
            for obj in objs:
                type = obj.find('name').text
                bndbox = obj.find('bndbox')
                if type == 'odometer':
                    x_odo_min = bndbox.find('xmin').text
                    y_odo_min = bndbox.find('ymin').text
                    x_odo_max = bndbox.find('xmax').text
                    y_odo_max = bndbox.find('ymax').text
                elif type == 't_odometer':
                    x_t_odo_min = bndbox.find('xmin').text
                    y_t_odo_min = bndbox.find('ymin').text
                    x_t_odo_max = bndbox.find('xmax').text
                    y_t_odo_max = bndbox.find('ymax').text
            # create dummy file
            dummy_file = create_dummy_file(image_file)
            for i in range(0,5):
                new_file_id = 'copy_{}_{}'.format(file.split('.')[0], i)
                new_file_name = new_file_id + '.jpg'
                new_im, odo_text, t_odo_text = simulate_odo(dummy_file,
                                                            (int(x_odo_min), int(y_odo_min)),
                                                            (int(x_t_odo_min), int(y_t_odo_min)))
                new_im.save(os.path.join('JPEGImages', new_file_name), "JPEG")
                # update Annotations folder
                add_annotations[new_file_id] = xml_template.format(
                    fileid=new_file_id,
                    filename=new_file_name,
                    width=width,
                    height=height,
                    depth=depth,
                    x_odo_min=x_odo_min,
                    y_odo_min=y_odo_min,
                    x_odo_max=x_odo_max,
                    y_odo_max=y_odo_max,
                    odo_val=odo_text,
                    x_t_odo_min=x_t_odo_min,
                    y_t_odo_min=y_t_odo_min,
                    x_t_odo_max=x_t_odo_max,
                    y_t_odo_max=y_t_odo_max,
                    t_odo_val="{0:.1f}".format(t_odo_text)
                )
                # Add entry in odoemter and t_odometer
                with open(os.path.join('ImageSets', 'Main', 'odometer_train.txt'), 'a+') as odo_file:
                    odo_file.writelines('\n' + new_file_id + ' 1')
                with open(os.path.join('ImageSets', 'Main', 't_odometer_train.txt'), 'a+') as odo_file:
                    odo_file.writelines('\n' + new_file_id + ' 1')
            # delete dummy file
            if os.path.exists(dummy_file):
                os.remove(dummy_file)

        else:
          pass

    for key in add_annotations.keys():
        with open(os.path.join('Annotations', key+'.xml'), 'w+') as file:
            file.write(add_annotations[key])


def simulate_odo(filename, odo_box, t_odo_box):
    im = Image.open(filename)
    new_im = im.copy()
    odo_val = random.randint(2000, 200000)
    trip_odo = random.uniform(1,odo_val/4)
    draw = ImageDraw.Draw(new_im)
    font = ImageFont.truetype("digital-7.ttf", 28)

    # ODO
    draw.text(odo_box,str(odo_val),(255,255,255),font=font)

    # TRIP ODO
    draw.text(t_odo_box,"{0:.1f}".format(trip_odo),(255,255,255),font=font)
    #     create_annotation(odoval, t_odoval)
    return new_im, odo_val, trip_odo


create_annotation()

print('Hello')