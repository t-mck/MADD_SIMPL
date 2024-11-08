import cv2
import numpy as np
import json
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import shutil

class SIMPL_Data_Annotater:
    def __init__(self,
                 image_files,
                 output_json='mask_data.json',
                 output_truth_yaml=False,
                 class_labels=None,
                 save_bboxes=True,
                 save_segmentation_masks=True,
                 save_segmentation_masks_inner_points=False,
                 normalize_annotation_points_to_0_1=True
        ):
        self.image_files = image_files
        self.output_json = output_json
        self.output_truth_yaml = output_truth_yaml
        self.mask_data = {}      # Dictionary to store masks for each image
        self.points = []         # Current points of the polygon being drawn
        self.img = None          # Current image being processed
        self.img_copy = None     # Copy of the image for resetting after each mask
        self.img_original = None # Original image to reset when moving to next image
        self.current_image_file = None  # Name of the current image file
        self.current_image_height = None
        self.current_image_width = None
        self.current_class = None  # Current object class (0-9)
        self.save_bboxes = save_bboxes
        self.save_segmentation_masks = save_segmentation_masks
        self.save_segmentation_masks_inner_points = save_segmentation_masks_inner_points
        self.normalize_annotation_points_to_0_1 = normalize_annotation_points_to_0_1
        self.class_labels = class_labels
        if self.class_labels is None:
            raise ValueError("class_labels not defined, and must be provided in a list ['target_0', 'target_1', ...]")

        # Initialize Tkinter window
        self.root = tk.Tk()
        rtitle = 'SIMPL Data Annotater. RECORDING: Class labels'
        if self.save_segmentation_masks:
            if self.save_segmentation_masks_inner_points:
                rtitle += ', Segmentation Masks Boundary & Inner Points'
            else:
                rtitle += ', Segmentation Masks Boundary Points'
        if self.save_bboxes:
            rtitle += ', Bounding Boxes'

        self.root.title(rtitle)

        # Create a frame for the buttons
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(side=tk.TOP, fill=tk.X)

        # Create class selection buttons (0-9)
        self.class_buttons = []
        for i in range(len(class_labels)):
            btn = tk.Button(self.button_frame,
                            text=f'{class_labels[i]} ({i})',#str(i),
                            command=lambda i=i: self.set_class(i),
                            bg='cornflowerblue',
                            fg='white',
                            activebackground='blue')
            btn.pack(side=tk.LEFT)
            self.class_buttons.append(btn)

        # Create other control buttons
        self.save_button = tk.Button(self.button_frame, text="Save Mask (s)", command=self.save_mask, activebackground='dimgrey')
        self.save_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(self.button_frame, text="Reset Mask (r)", command=self.reset_mask, activebackground='dimgrey')
        self.reset_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(self.button_frame, text="Next Image (n)", command=self.next_image, activebackground='dimgrey')
        self.next_button.pack(side=tk.LEFT)

        self.exit_button = tk.Button(self.button_frame, text="Exit (Esc)", command=self.exit_tool, activebackground='dimgrey')
        self.exit_button.pack(side=tk.LEFT)

        # Create a canvas for image display
        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bind mouse and keyboard events
        self.canvas.bind("<Button-1>", self.left_click)    # Left mouse button
        self.canvas.bind("<Button-3>", self.right_click)   # Right mouse button
        self.root.bind("<Key>", self.key_press)            # Key press events

        # Load the first image
        self.image_index = 0
        self.load_image()

        # Start the Tkinter main loop
        self.root.mainloop()

    def set_class(self, class_id):
        """
        Sets the current object class.
        """
        self.current_class = class_id
        print(f"Current class set to: {class_id}")

    def load_image(self):
        """
        Loads the current image and sets up the canvas.
        """
        if self.image_index >= len(self.image_files):
            messagebox.showinfo("Info", "No more images to process.")
            self.save_masks()
            self.root.destroy()
            return

        self.current_image_file = self.image_files[self.image_index]
        self.img = cv2.imread(self.current_image_file)
        if self.img is None:
            print(f"Failed to load {self.current_image_file}")
            self.image_index += 1
            self.load_image()
            return
        self.current_image_height = self.img.shape[0]
        self.current_image_width = self.img.shape[1]

        self.img_original = self.img.copy()
        self.img_copy = self.img.copy()
        self.masks = []  # List of masks for this image
        self.points = []

        # Convert the image to PIL format
        self.display_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.display_image = Image.fromarray(self.display_image)
        self.photo = ImageTk.PhotoImage(image=self.display_image)

        # Set canvas size and display the image
        self.canvas.config(width=self.photo.width(), height=self.photo.height())
        self.canvas_image = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        print(f"\nProcessing {self.current_image_file}.")
        print("Instructions:")
        print(" - Left-click to add points to the polygon.")
        print(" - Right-click to close the polygon.")
        print(" - Select object class (0-9) or press the digit key (e.g., '0', '1', etc.) before saving.")
        print(" - Click 'Save Mask (s)' or press 's' to save the mask.")
        print(" - Click 'Reset Mask (r)' or press 'r' to reset the current mask.")
        print(" - Click 'Next Image (n)' or press 'n' to move to the next image.")
        print(" - Click 'Exit (Esc)' or press 'Esc' to exit.")

    def update_canvas(self):
        """
        Updates the canvas with the current image.
        """
        self.display_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.display_image = Image.fromarray(self.display_image)
        self.photo = ImageTk.PhotoImage(image=self.display_image)
        self.canvas.itemconfig(self.canvas_image, image=self.photo)

    def left_click(self, event):
        """
        Handles left mouse button click events.
        """
        x, y = event.x, event.y
        self.points.append((x, y))
        # Draw a small circle at the point
        cv2.circle(self.img, (x, y), 2, (0, 255, 0), -1)
        # Draw lines connecting the points
        if len(self.points) > 1:
            cv2.line(self.img, self.points[-2], self.points[-1], (0, 255, 0), 1)
        self.update_canvas()

    def right_click(self, event):
        """
        Handles right mouse button click events.
        """
        if len(self.points) > 2:
            # Close the polygon by connecting the last point to the first
            cv2.line(self.img, self.points[-1], self.points[0], (0, 255, 0), 1)
            self.update_canvas()
            print("Polygon completed. Select object class and press 's' to save or 'r' to reset.")
        else:
            print("Need at least 3 points to form a polygon.")

    def key_press(self, event):
        """
        Handles key press events.
        """
        if event.char == 'n':
            self.next_image()
        elif event.char == 'r':
            self.reset_mask()
        elif event.char == 's':
            self.save_mask()
        elif event.char == '\x1b':  # Escape key
            self.exit_tool()
        elif event.char.isdigit():
            self.set_class(int(event.char))

    def save_mask(self):
        """
        Saves the current mask with the selected object class.
        """
        if len(self.points) > 2:
            if self.current_class is not None:
                mask_info = {
                    'class': self.current_class
                }
                if self.save_bboxes:
                    mask_info['bbox'] = self.polygon_to_bbox(self.points.copy())

                if self.save_segmentation_masks:
                    if self.save_segmentation_masks_inner_points:
                        mask_info['segmentation_mask_points'] = self.get_polygon_all_points(self.points.copy())
                    else:
                        if self.normalize_annotation_points_to_0_1:
                                mask_info['segmentation_mask_points'] =  self.normalize_annotation_points_to_0_1_for_list(self.points.copy())
                        else:
                            mask_info['segmentation_mask_points'] = self.points.copy()

                if (not self.save_segmentation_masks) and (not self.save_bboxes):
                    mask_info['segmentation_mask_points'] = self.points.copy()

                self.masks.append(mask_info)
                print(f"Mask saved with class {self.current_class} and {len(self.points)} points.")
                # Fill the saved mask with a color to visualize
                cv2.fillPoly(self.img_copy, [np.array(self.points, dtype=np.int32)], (0, 0, 255))
                self.img = self.img_copy.copy()
                self.update_canvas()
                # Reset for the next mask
                self.points = []
                self.current_class = None
            else:
                messagebox.showwarning("Warning", "Please select an object class before saving.")
        else:
            print("Mask not saved. Need at least 3 points.")

    def reset_mask(self):
        """
        Resets the current mask being drawn.
        """
        self.img = self.img_copy.copy()
        self.points = []
        self.update_canvas()
        print("Current mask reset.")

    def next_image(self):
        """
        Moves to the next image in the list.
        """
        self.mask_data[self.current_image_file] = self.masks
        self.image_index += 1
        self.load_image()

    def exit_tool(self):
        """
        Exits the tool and saves the mask data.
        """
        self.mask_data[self.current_image_file] = self.masks
        self.save_masks()
        self.root.destroy()

    def save_masks(self):
        """
        Saves the collected mask data to a JSON file.
        """
        # Convert the mask data to a serializable format
        serializable_mask_data = {}
        for image_file, masks in self.mask_data.items():
            serializable_masks = []
            for mask_info in masks:
                serializable_mask = {
                    'class': mask_info['class']
                }
                if self.save_bboxes:
                    serializable_mask['bbox'] =  mask_info['bbox']
                if self.save_segmentation_masks:
                    serializable_mask['segmentation_mask_points'] = [list(point) for point in mask_info['segmentation_mask_points']]
                if (not self.save_segmentation_masks) and (not self.save_bboxes):
                    serializable_mask['segmentation_mask_points'] = [list(point) for point in
                                                                     mask_info['segmentation_mask_points']]

                serializable_masks.append(serializable_mask)
            serializable_mask_data[image_file] = serializable_masks

        # Save the mask data to a JSON file
        with open(self.output_json, 'w') as f:
            json.dump(serializable_mask_data, f)

        print(f"\nMask data saved to '{self.output_json}'.")

        if self.output_truth_yaml:
            self.make_truth_yaml(truth_dict = serializable_mask_data)

    def polygon_to_bbox(self,
                        points):
        """
        Given a list of (x, y) points defining a polygon, compute the bounding box
        with x center, y center, x width, and y width.

        Parameters:
        - points: List of (x, y) tuples or lists representing the polygon vertices.

        Returns:
        - bbox: A tuple (x_center, y_center, x_width, y_width)
        """
        if not points:
            raise ValueError("The points list is empty.")

        # Extract x and y coordinates
        x_coords = [point[0] for point in points]
        y_coords = [point[1] for point in points]

        # Compute min and max coordinates
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Compute center and width/height
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        x_width = x_max - x_min
        y_width = y_max - y_min

        if self.normalize_annotation_points_to_0_1:
            x_center = x_center/self.current_image_width
            y_center = y_center / self.current_image_height
            x_width = x_width / self.current_image_width
            y_width = y_width / self.current_image_height

        return (x_center, y_center, x_width, y_width)

    def get_polygon_all_points(self,
                               polygon,
                               image_shape=None):
        """
        Given a list of (x, y) points defining the boundary of a polygon,
        return the set of all integer-coordinate points inside the polygon,
        including the boundary points.

        Parameters:
        - polygon: List of (x, y) tuples or lists representing the polygon vertices.
        - image_shape: Optional tuple (height, width) defining the size of the image canvas.
                       If not provided, it will be computed based on the polygon extents.

        Returns:
        - points_inside: List of (x, y) tuples representing all the points inside the polygon.
        """
        # Ensure the polygon has at least 3 points
        if len(polygon) < 3:
            raise ValueError("Polygon must have at least 3 points.")

        # Extract x and y coordinates
        x_coords = [int(point[0]) for point in polygon]
        y_coords = [int(point[1]) for point in polygon]

        # Compute the bounding box of the polygon if image_shape is not provided
        if image_shape is None:
            x_min = min(x_coords)
            x_max = max(x_coords)
            y_min = min(y_coords)
            y_max = max(y_coords)
            width = x_max + 1
            height = y_max + 1
        else:
            height, width = image_shape

        # Create a blank mask image
        mask = np.zeros((height, width), dtype=np.uint8)

        # Convert polygon points to a NumPy array of integer coordinates
        pts = np.array([polygon], dtype=np.int32)

        # Fill the polygon on the mask image
        cv2.fillPoly(mask, pts, color=1)

        # Find the indices (y, x) where mask is 1
        y_indices, x_indices = np.where(mask == 1)

        # Combine x and y indices into a list of (x, y) tuples
        points_inside = list(zip(x_indices.tolist(), y_indices.tolist()))

        if self.normalize_annotation_points_to_0_1:
            points_inside = self.normalize_annotation_points_to_0_1_for_list(points_inside)

        return points_inside

    def normalize_annotation_points_to_0_1_for_list(self,list_of_points):
        for p in range(len(list_of_points)):
            list_of_points[p][0] = list_of_points[p][0]/self.current_image_width
            list_of_points[p][1] = list_of_points[p][1]/self.current_image_height
        return list_of_points

    # def make_truth_yaml(self, truth_dict, class_labels, dataset_name, dataset_path):
    #     '''
    #     # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
    #     path: ../datasets/coco8 # dataset root dir
    #     train: images/train # train images (relative to 'path') 4 images
    #     val: images/val # val images (relative to 'path') 4 images
    #     test: # test images (optional)
    #
    #     # Classes (80 COCO classes)
    #     names:
    #         0: person
    #         1: bicycle
    #         2: car
    #         # ...
    #         77: teddy bear
    #         78: hair drier
    #         79: toothbrush
    #     :return:
    #     '''
    #     path = f'path: ./{dataset_name}\n'
    #     test = f'test: images/test\n'
    #     names_str = 'names:\n'
    #     for c in range(len(class_labels)):
    #         names_str += f'  {c}: {class_labels[c]}\n'
    #
    #     file_contents = path + test + names_str
    #     with open(dataset_path + '/data.yaml', 'w') as f:
    #         f.write(file_contents)
    #
    #     for anno_image in truth_dict:
    #         # Copy Image
    #         image_current_path = truth_dict[anno_image]
    #         new_file_name_path = dir_to_copy_to + f'/{anno_image}.png'
    #         shutil.copy(image_current_path, new_file_name_path)
    #
    #         # Write annotation files
    #         if self.save_bboxes:
    #             image_labels = truth_dict['bbox']
    #             new_file_name_path = dir_to_copy_to + f'/{anno_image}.txt'
    #             with open(new_file_name_path, 'w') as f:
    #                 for label in image_labels:
    #                     f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")
    #
    #         if self.save_segmentation_masks:
    #             image_labels = truth_dict['segmentation_mask_points']
    #             new_file_name_path = dir_to_copy_to + f'/{anno_image}.txt'
    #             with open(new_file_name_path, 'w') as f:
    #                 for label in image_labels:
    #                     line = f'{label[0]}'
    #                     for n in range(1,len(label)):
    #                         line += f' {label[n]}'
    #                     f.write(f"{line}\n")




def main():
    # Replace with your list of image filenames
    image_files = ["C:/Users/tm-pc-win/Pictures/CVN_AC_Yokosuka.png",
                   "C:/Users/tm-pc-win/Pictures/CVN_AC_Yokosuka.png"]  # Replace with actual image paths
    SIMPL_Data_Annotater(image_files=image_files,
                         output_json='mask_data_0.json',
                         output_truth_yaml=True,
                         class_labels=
                         [
                            "SUV",
                            "Truck",
                            "AC",
                            "Destroyer",
                            "Cruiser",
                            "Submarine",
                            "Non-Target"
                         ],
                         save_bboxes=True,
                         save_segmentation_masks=True,
                         save_segmentation_masks_inner_points=False,
                         normalize_annotation_points_to_0_1=False)

if __name__ == "__main__":
    main()
