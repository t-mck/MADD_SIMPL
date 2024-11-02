from PIL import Image, ImageDraw, ImageFont, ImageFile
import os
from pathlib import Path
import builtins
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from SIMPL_TruthJSON import SIMPL_TruthJSON
import cv2

class SIMPL_TruthPlotter:
    def __init__(self):
        pass

    @staticmethod
    def get_all_files(directory_path):
        # List all files in the given directory
        filez = os.listdir(directory_path)
        files = []
        for f in filez:
            files.append(repr(f)[1:-1])
        return files

    def add_bboxes_to_image(self,
                            img_path,
                            bboxes,
                            bbox_color='blue'):
        # Open the original image
        file_path = Path(img_path)
        image = Image.open(file_path).convert("RGBA")

        # Create a drawing context
        draw = ImageDraw.Draw(image, "RGBA")

        for bbox in bboxes:
            # Define the label box (top portion)
            box_coords = [bbox[1], bbox[0], bbox[3], bbox[2]]
            # Add a semi-transparent black rectangle as the label box
            draw.rectangle(box_coords, fill=None, outline=bbox_color, width=3)

        # Save the modified image with the same name
        image.save(img_path)

    def get_transparent_mask(self, img_height, img_width):
        transparent_value = (255,255,255)
        img_mat = []
        for y in range(0, img_height):
            row = []
            for x in range(0, img_width):
                row.append(transparent_value)
            img_mat.append(row)

        transparent_mask = np.array(img_mat)
        return transparent_mask

    def add_segmentation_masks_to_image(self,
                                       file_path,
                                       segmentation_masks,
                                       mask_color='blue',
                                        object_label='MRAP'):
        # Open the original image
        image = Image.open(file_path).convert("RGBA")
        image_array = np.array(image)
        plt.imshow(image_array)

        for sm in segmentation_masks:
            x = []
            y = []
            for pixel in segmentation_masks[sm]:
                y.append(pixel[0])
                x.append(pixel[1])
            obj_lab_x = int(np.mean(np.array(x)))
            obj_lab_y = int(np.min(np.array(y))) - int(608*0.01)
            plt.plot(x, y, color=mask_color, alpha=0.3)
            plt.text(x=obj_lab_x,y=obj_lab_y,s=object_label, color=mask_color)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

    def draw_truth_on_images(self,
                             truth_dat):
        directory_path = r'C:\Users\tm-pc-win\Documents\AMLL\ONR_Code35\CE_Output_Images\test'

        files = self.get_all_files(directory_path=directory_path)
        for f in files:
            pitch_start = f.find('_pitch')
            yaw_start = f.find("_yaw")
            dto_start = f.find("_distToOrig")
            dto_end = f.find(".png")

            pitch = f[pitch_start + 6:yaw_start]
            yaw = f[yaw_start + 4:dto_start]
            dto = f[dto_start + 11:dto_end]
            lab = f'{pitch}_{yaw}_{dto}'

            bboxes = truth_dat[lab]

            img_path = directory_path + "/" + f
            self.add_bboxes_to_image(img_path, bboxes)


def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    stj = SIMPL_TruthJSON()
    truth_dict = stj.get_truth_dict('MRAP')
    image_path = f'C:\\Users\\tm-pc-win\\Documents\\CityEngine\\Default Workspace\\MADD_SIMPL_v0\\images\\SIMPL_images\\color_all_images_step182.4\\color_SIMPL_xview_background_sd1663_1_pitch90_yaw0_distToOrig250.png'
    file_path = Path(image_path)

    stp = SIMPL_TruthPlotter()
    stp.add_segmentation_masks_to_image(file_path=file_path,
                                        segmentation_masks=truth_dict['color_SIMPL_xview_background_sd1663_1_pitch90_yaw0_distToOrig250.png']['segmentation_masks'])
    # truth_dat = get_truth_labels()
    # draw_truth_on_images(truth_dat)
    # banner_images()


if __name__ == "__main__":
    main()
