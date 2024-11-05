import cv2
import numpy as np
import json
from tkinter import filedialog

class SIMPL_TruthSegmentationRecorder:
    '''
    This class was made with the assistance of GPTo1-Preview
    '''
    def __init__(self, image_files, output_json='mask_data.json'):
        self.image_files = image_files
        self.output_json = output_json
        self.mask_data = {}  # Dictionary to store masks for each image
        self.points = []     # Current points of the polygon being drawn
        self.img = None      # Current image being processed
        self.img_copy = None # Copy of the image for resetting after each mask
        self.img_original = None  # Original image to reset when moving to next image
        self.current_image_file = None  # Name of the current image file

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function to handle drawing polygons on the image.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left-click to add a point
            self.points.append((x, y))
            # Draw a small circle at the point
            cv2.circle(self.img, (x, y), 2, (0, 255, 0), -1)
            # Draw lines connecting the points
            if len(self.points) > 1:
                cv2.line(self.img, self.points[-2], self.points[-1], (0, 255, 0), 1)
            cv2.imshow('image', self.img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right-click to close the polygon
            if len(self.points) > 2:
                # Close the polygon by connecting the last point to the first
                cv2.line(self.img, self.points[-1], self.points[0], (0, 255, 0), 1)
                cv2.imshow('image', self.img)
                print("Polygon completed. Press 's' to save or 'r' to reset.")
            else:
                print("Need at least 3 points to form a polygon.")

    def process_images(self):
        """
        Main method to process each image in the list.
        """
        # for image_file in self.image_files:
        #     self.current_image_file = image_file
        for ifl in image_files:
            image_file = filedialog.askopenfilename()
            self.img = cv2.imread(image_file)
            if self.img is None:
                print(f"Failed to load {image_file}")
                continue

            self.img_original = self.img.copy()
            self.img_copy = self.img.copy()
            masks = []
            self.points = []

            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.mouse_callback)

            print(f"\nProcessing {image_file}.")
            print("Instructions:")
            print(" - Left-click to add points to the polygon.")
            print(" - Right-click to close the polygon.")
            print(" - Press 's' to save the mask.")
            print(" - Press 'r' to reset the current mask.")
            print(" - Press 'n' to move to the next image.")
            print(" - Press 'Esc' to exit.")

            while True:
                cv2.imshow('image', self.img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('n'):
                    # Move to the next image
                    print("Moving to the next image...")
                    self.img = self.img_original.copy()
                    break
                elif key == ord('r'):
                    # Reset the current mask
                    self.img = self.img_copy.copy()
                    self.points = []
                    print("Current mask reset.")
                elif key == ord('s'):
                    # Save the current mask
                    if len(self.points) > 2:
                        masks.append(self.points.copy())
                        print(f"Mask saved with {len(self.points)} points.")
                        # Fill the saved mask with a color to visualize
                        cv2.fillPoly(self.img_copy, [np.array(self.points)], (0, 0, 255))
                        self.img = self.img_copy.copy()
                    else:
                        print("Mask not saved. Need at least 3 points.")
                    # Reset for the next mask
                    self.points = []
                elif key == 27:
                    # Esc key pressed
                    print("Exiting...")
                    break

            self.mask_data[image_file] = masks
            if key == 27:
                break

        cv2.destroyAllWindows()
        self.save_masks()

    def save_masks(self):
        """
        Saves the collected mask data to a JSON file.
        """
        # Convert the mask data to a serializable format
        serializable_mask_data = {}
        for image_file, masks in self.mask_data.items():
            serializable_masks = []
            for mask in masks:
                serializable_mask = [list(point) for point in mask]
                serializable_masks.append(serializable_mask)
            serializable_mask_data[image_file] = serializable_masks

        # Save the mask data to a JSON file
        with open(self.output_json, 'w') as f:
            json.dump(serializable_mask_data, f)

        print(f"\nMask data saved to '{self.output_json}'.")

if __name__ == "__main__":
    # Replace with your list of image filenames
    image_files = ['image1.jpg', 'image2.jpg']
    segmentation_tool = SIMPL_TruthSegmentationRecorder(image_files)
    segmentation_tool.process_images()
