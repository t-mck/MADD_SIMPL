from idlelib.debugobj_r import remote_object_tree_item

from PIL import Image
import os
from pathlib import Path
import numpy as np
from skimage.measure import label, regionprops
import json
from PIL import Image, ImageDraw, ImageFont, ImageFile
import shutil
import random

class SIMPL_TruthYAML:
    def __init__(self):
        pass

    def threshold_colors(self,
                         image_np_arr,
                         pitch,
                         pitch_threshold = 10,
                         object_pixel_value = 0,
                         background_pixel_value = 255):
        '''

        pitch values less than the pictch_threshold show the sky in CityEngine, which has a blue color. These need to
        be converted to white to properly obtain truth pixels. Additionally, when the image is viewed from the top down
        the 3dObjects can have a shiny quality which makes some of their faces have an off-white color, this needs to be
        converted to the overall object truth values to get good bounding boxes or segmentation masks.

        :param image_np_arr:
        :param pitch:
        :param pitch_threshold:
        :param object_pixel_value:
        :param background_pixel_value:
        :return:
        '''
        if pitch < pitch_threshold:
            image_np_arr[image_np_arr > object_pixel_value] = background_pixel_value
        else:
            image_np_arr[image_np_arr < background_pixel_value] = object_pixel_value

        return image_np_arr

    def convert_image_to_numpy(self,
                               image,
                               pitch,
                               object_pixel_value=0,
                               background_pixel_value=255):
        image = image.convert("L")
        img_arr = np.array(image)
        img_arr = self.threshold_colors(img_arr,
                                        pitch=pitch,
                                        object_pixel_value=object_pixel_value,
                                        background_pixel_value=background_pixel_value)
        return img_arr

    def get_bboxes(self,
                   image,
                   pitch,
                   bbox_area_threshold=0.25,
                   object_pixel_value=0,
                   background_pixel_value=255):
        img_arr = self.convert_image_to_numpy(image,
                                              pitch=pitch,
                                              object_pixel_value=object_pixel_value,
                                              background_pixel_value=background_pixel_value)
        labeled_image = label(img_arr, background=background_pixel_value, connectivity=2) # connectivity 2 looks for all 8 adjacent pixels, connectivity 1 just looks for left, right, top, and bottom

        regions = regionprops(labeled_image)

        max_box_area = 0
        for region in regions:
            area = region.area_bbox
            if area > max_box_area:
                max_box_area = area

        #bboxes = []
        bboxes = {}
        i = 0
        for region in regions:
            area = region.area_bbox
            if area >= max_box_area * bbox_area_threshold:
                obj_name = f'obj_num_{i}'
                bboxes[obj_name] = region.bbox
                #bboxes.append(region.bbox)
                i+=1

        return bboxes

    def get_object_segmentation_mask_from_whole_image(self, image, pitch):
        '''
        This function is only useful if there is only one object in the image, or if the segmentation will be
        narrowed down using a bounding box

        :param image:
        :param pitch:
        :return:
        '''
        img_arr = self.convert_image_to_numpy(image, pitch=pitch)
        object_pixels = np.where(img_arr < 255)
        return object_pixels

    def get_object_segmentation_mask_from_bbox(self, obp_x, obp_y, pitch, bbox):

        object_segmentation_pixels = []
        for i in range(0, len(obp_x)):
            pixel = (obp_x[i], obp_y[i])
            if (pixel[0] >= bbox[0]) and (pixel[0] <= bbox[2]):  # inside x range of bbox
                if (pixel[1] >= bbox[1]) and (pixel[1] <= bbox[3]):  # inside y range of bbox
                    object_segmentation_pixels.append(pixel)

        return object_segmentation_pixels

    def get_object_segmentation_mask_from_bboxes(self,
                                                 image,
                                                 pitch,
                                                 bboxes):
        seg_masks = {}
        i = 0
        object_bbox_pixels = self.get_object_segmentation_mask_from_whole_image(image, pitch)
        obp_x = object_bbox_pixels[0]
        obp_y = object_bbox_pixels[1]
        for box in bboxes:
            object_bbox_pixels = self.get_object_segmentation_mask_from_bbox(obp_x, obp_y, pitch, bboxes[box])
            obj_num = f'obj_num_{i}'
            seg_masks[obj_num] = object_bbox_pixels
            i+=1

        return seg_masks

    @staticmethod
    def get_all_files(directory_path):
        # List all files in the given directory
        filez = os.listdir(directory_path)
        files = []
        for f in filez:
            files.append(repr(f)[1:-1])
        return files

    def get_ula_labels(self,
                       bboxes,
                       class_number,
                       img_size):
        ula_labels = []
        for b in bboxes:
            bbox = bboxes[b]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            x_center = int((bbox[2] - bbox[0])/2)
            y_center = int((bbox[3] - bbox[1])/2)

            ula_width = width/img_size
            ula_height = height/img_size
            ula_x_center = x_center/img_size
            ula_y_center = y_center/img_size

            ula_label = [class_number, ula_x_center, ula_y_center, ula_width, ula_height]
            ula_labels.append(ula_label)

        return ula_labels

    def get_truth_dict(self,
                       object_label,
                       directory_path = r'C:\Users\tm-pc-win\Documents\AMLL\ONR_Code35\CE_Output_Images\test_truth'):

        files = self.get_all_files(directory_path=directory_path)
        truth_label_info = []
        image_number = 0
        for f in files:
            pitch_start = f.find('_pitch')
            yaw_start = f.find("_yaw")
            dto_start = f.find("_distToOrig")
            dto_end = f.find(".png")

            pitch = f[pitch_start + 6:yaw_start]
            yaw = f[yaw_start + 4:dto_start]
            dto = f[dto_start + 11:dto_end]
            lab = f'{f}'
            origin_information = {
                'pitch': pitch,
                'yaw': yaw,
                'camera_distance_to_origin': dto,
            }

            img_path = directory_path + "/" + f
            file_path = Path(img_path)
            image = Image.open(file_path).convert("RGBA")
            bboxes = self.get_bboxes(image, pitch=int(pitch))
            ula_labels = self.get_ula_labels(bboxes, class_number=object_label, img_size=640)
            fnp = directory_path + '/' + f
            info_dict = {
                'file': fnp.replace('annos', 'images'),
                'labels': ula_labels
            }
            truth_label_info.append(info_dict)
            image_number += 1

        return truth_label_info

    def make_yaml_file_structure(self,
                                 dataset_path,
                                 yaml_subs = ['images', 'labels'],
                                 sub_subs = ['train', 'val', 'test']):
        for ys in yaml_subs:
            dataset_path_ys = dataset_path + f'{ys}/'
            if not os.path.exists(dataset_path_ys):
                os.mkdir(dataset_path_ys)
            for ss in sub_subs:
                dataset_path_s = dataset_path_ys + f'{ss}/'
                if not os.path.exists(dataset_path_s):
                    os.mkdir(dataset_path_s)


    def copy_images_to_yaml_file_structure(self,
                                           dir_to_copy_to,
                                           image_set_to_copy,
                                           truth_label_info):
        for i in image_set_to_copy:#range(image_set_to_copy[0], image_set_to_copy[1]):
            image_current_path = truth_label_info[i]['file']
            new_file_name_path = dir_to_copy_to +  f'/{i}.png'
            shutil.copy(image_current_path, new_file_name_path)

    def write_labels_to_yaml_file_structure(self,
                                            dir_to_copy_to,
                                            image_set_to_copy,
                                            truth_label_info):
        for i in image_set_to_copy:#range(image_set_to_copy[0], image_set_to_copy[1]):
            image_labels = truth_label_info[i]['labels']
            new_file_name_path = dir_to_copy_to +  f'/{i}.txt'
            with open(new_file_name_path, 'w') as f:
                for label in image_labels:
                    f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")

    def make_yaml_file(self,
                       dataset_path,
                       dataset_name,
                       class_numbers_and_names):
        '''
        # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
        path: ../datasets/coco8 # dataset root dir
        train: images/train # train images (relative to 'path') 4 images
        val: images/val # val images (relative to 'path') 4 images
        test: # test images (optional)

        # Classes (80 COCO classes)
        names:
            0: person
            1: bicycle
            2: car
            # ...
            77: teddy bear
            78: hair drier
            79: toothbrush
        :return:
        '''
        path = f'path: ./{dataset_name}\n'
        train = f'train: images/train\n'
        val = f'val: images/val\n'
        test = f'test: images/test\n'
        names_str = 'names:\n'
        for cnn in class_numbers_and_names:
            names_str += f'  {cnn[0]}: {cnn[1]}\n'

        file_contents = path + train + val + test + names_str
        with open(dataset_path + '/data.yaml', 'w') as f:
            f.write(file_contents)

    def make_training_yaml(self,
                           truth_label_info,
                           train=0.8,
                           validation=0.1,
                           test=0.1,
                           dataset_path="C:\\Users\\tm-pc-win\\Documents\\AMLL\\Synthetic_Data_Generation\\datasets\\MADD_overhead_dsiac_test\\",
                           object_names_classes=[],
                           dataset_name='MADD_overhead_dsiac_test'
                           ):

        self.make_yaml_file_structure(dataset_path=dataset_path)

        num_images = len(truth_label_info)

        # train_range = (0, int(num_images * train))
        # validation_range = (train_range[1], int(train_range[1] + num_images * validation))
        # test_range = (validation_range[1], int(validation_range[1] + num_images * test))

        num_train = int(num_images * train)
        num_validation = int(num_images * validation)
        num_test = int(num_images * test)

        numbers = list(range(0, num_images))

        # Shuffle the list randomly
        random.shuffle(numbers)

        # Split the list into three equal parts
        train_split = numbers[:num_train]
        val_split = numbers[num_train:num_train+num_validation]
        test_split = numbers[num_train+num_validation:num_train+num_validation+num_test]

        self.copy_images_to_yaml_file_structure(dir_to_copy_to=dataset_path+'images/train/',image_set_to_copy=train_split,truth_label_info=truth_label_info)
        self.copy_images_to_yaml_file_structure(dir_to_copy_to=dataset_path+'images/val/',image_set_to_copy=val_split,truth_label_info=truth_label_info)
        self.copy_images_to_yaml_file_structure(dir_to_copy_to=dataset_path+'images/test/',image_set_to_copy=test_split,truth_label_info=truth_label_info)

        self.write_labels_to_yaml_file_structure(dir_to_copy_to=dataset_path+'labels/train/',image_set_to_copy=train_split,truth_label_info=truth_label_info)
        self.write_labels_to_yaml_file_structure(dir_to_copy_to=dataset_path+'labels/val/',image_set_to_copy=val_split,truth_label_info=truth_label_info)
        self.write_labels_to_yaml_file_structure(dir_to_copy_to=dataset_path+'labels/test/',image_set_to_copy=test_split,truth_label_info=truth_label_info)

        self.make_yaml_file(dataset_path=dataset_path,
                            dataset_name=dataset_name,
                            class_numbers_and_names=object_names_classes)

def main():
    parent_dir = "C:\\Users\\tm-pc-win\\Documents\\AMLL\\Synthetic_Data_Generation\\datasets\\prep\\"
    stj = SIMPL_TruthYAML()
    object_classes = {
        0: "Pickup",
        1: "SUV",
        2: "BTR82",
        3: "BRDM",
        4: "BMP2",
        5: "T72",
        6: "ZSU",
        7: "2S3",
        8:  "MTLB"
    }
    object_names_classes = [
        ["0", "Pickup"],
        ["1", "SUV"],
        ["2", "BTR82"],
        ["3", "BRDM"],
        ["4", "BMP2"],
        ["5", "T72"],
        ["6", "ZSU"],
        ["7", "2S3"],
        ["8",  "MTLB"]
    ]
    tli_list = []
    for cl in object_classes.keys():
        cl_num = int(cl)
        cl_name = object_classes[cl]
        truth_label_info = stj.get_truth_dict(object_label=cl_num,
                                              directory_path=parent_dir + f'color_{cl_name}_all_annos_step608.0')
        for i in truth_label_info:
            tli_list.append(i)

    dataset_path = "C:\\Users\\tm-pc-win\\Documents\\AMLL\\Synthetic_Data_Generation\\datasets\\MADD_overhead_dsiac_test\\"
    stj.make_training_yaml(truth_label_info=tli_list,
                           train=0.8,
                           validation=0.1,
                           test=0.1,
                           dataset_path=dataset_path,
                           object_names_classes=object_names_classes,
                           dataset_name='MADD_overhead_dsiac_test')

if __name__ == '__main__':
    main()
