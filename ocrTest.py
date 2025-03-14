import cv2
import torch
import easyocr


def ocrTest(image):
    # Initialize EasyOCR reader for English
    reader = easyocr.Reader(['en'])

    results = reader.readtext(image)

    # Draw bounding boxes and text on the image
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))

        #BOX
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

        cv2.putText(image, text, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        #filename = "FinalTest.png"
        #cv2.imwrite(filename,  image) 

    return image, results