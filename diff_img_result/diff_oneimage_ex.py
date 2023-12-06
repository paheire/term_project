import cv2
import os

def resize_image(input_path, output_path, target_size=(800, 600)):
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"Error: Unable to read image from '{input_path}'.")
        return

    resized_img = cv2.resize(img, target_size, cv2.INTER_CUBIC)
    cv2.imwrite(output_path, resized_img)

def process_image(input_path):
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Unable to read image from '{input_path}'.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    split_line = int(width / 2)
    left_part = gray[:, :split_line]
    right_part = gray[:, split_line:]

    # Difference image
    diff = cv2.absdiff(left_part, right_part)
    _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Draw contours on the original image
    COLOR = (0, 200, 0)
    for cnt in contours:
        x, y, width, height = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + width, y + height), COLOR, 2)

    # Save result image
    result_path = os.path.splitext(input_path)[0] + '_result.jpg'
    cv2.imwrite(result_path, image)

    # Display the result
    resized_image = cv2.resize(image, (800, 600))
    cv2.imshow('difference', resized_image)
    cv2.waitKey(0)

# Input image path
input_image_path = r"C:\Users\admin\image\dif_pic.jpg"

# Resize and process the image
output_image_path = 'resized_img.jpg'
resize_image(input_image_path, output_image_path)
process_image(output_image_path)

# Close the window
cv2.destroyAllWindows()
