import cv2
import json
import os
from tkinter import filedialog

class SIMPL_TruthBBoxRecorderGUI:
    '''
    This class was made with the assistance of GPTo1-Preview
    '''
    def __init__(self, image_paths):
        # Initialize variables
        self.image_paths = image_paths
        self.boxes_dict = {}
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.rectangles = []
        self.img = None
        self.img_display = None

    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            print(f"Started drawing rectangle at ({self.ix}, {self.iy})")

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.img_display = self.img.copy()
                # Draw all existing rectangles
                for rect in self.rectangles:
                    cv2.rectangle(self.img_display, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
                # Draw the current rectangle
                cv2.rectangle(self.img_display, (self.ix, self.iy), (x, y), (0, 255, 0), 2)
                cv2.imshow('Image', self.img_display)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            # Save the rectangle coordinates
            self.rectangles.append((self.ix, self.iy, x, y))
            print(f"Finished drawing rectangle at ({x}, {y})")
            # Redraw the image with all rectangles
            self.img_display = self.img.copy()
            for rect in self.rectangles:
                cv2.rectangle(self.img_display, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow('Image', self.img_display)

    def process_images(self):
        for ip in self.image_paths:
            image_path = filedialog.askopenfilename()
            # Load the image
            self.img = cv2.imread(image_path)
            if self.img is None:
                print(f"Failed to load image {image_path}")
                continue

            print(f"Processing {image_path}...")
            self.rectangles = []
            self.img_display = self.img.copy()

            # Create a window and bind the function to window
            cv2.namedWindow('Image')
            cv2.setMouseCallback('Image', self.draw_rectangle)

            while True:
                cv2.imshow('Image', self.img_display)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('n'):  # Press 'n' to proceed to the next image
                    print("Moving to the next image...")
                    break
                elif k == ord('q'):  # Press 'q' to quit
                    print("Exiting...")
                    self.boxes_dict[os.path.basename(image_path)] = self.rectangles.copy()
                    cv2.destroyAllWindows()
                    return None

            # Save rectangles for the current image
            self.boxes_dict[os.path.basename(image_path)] = self.rectangles.copy()
            cv2.destroyAllWindows()

    def save_boxes(self, filename='boxes.json'):
        with open(filename, 'w') as f:
            json.dump(self.boxes_dict, f, indent=4)
            print(f"Box coordinates have been saved to {filename}")

def main():
    # List of image paths to process
    image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']  # Replace with your image paths

    strg = SIMPL_TruthBBoxRecorderGUI(image_paths)
    strg.process_images()
    strg.save_boxes('boxes.json')

if __name__ == '__main__':
    main()
