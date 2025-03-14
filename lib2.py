import cv2
import numpy as np
import requests
import csv

def correct_card_orientation(image_path, corners, contour):
    # Read the image
    img = image_path

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #for corner in corners:
        #print(corner)

        #cv2.circle(img, tuple(corner), 10, (0, 255, 0), -1)
        
    #print(corners)
    #print(corners[0])
    #print(corners[0][0][0])
    #print(corners)
        
    width_AD = np.sqrt(((corners[0][0] - corners[3][0]) ** 2) + ((corners[0][1] - corners[3][1]) ** 2))
    width_BC = np.sqrt(((corners[1][0] - corners[2][0]) ** 2) + ((corners[1][1] - corners[2][1]) ** 2))
    #print(width_AD)
    #print(width_BC)
    maxWidth = max(int(width_AD), int(width_BC))
    
    
    height_AB = np.sqrt(((corners[0][0] - corners[1][0]) ** 2) + ((corners[0][1] - corners[1][1]) ** 2))
    height_CD = np.sqrt(((corners[2][0] - corners[3][0]) ** 2) + ((corners[2][1] - corners[3][1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    dest_pts = np.float32([[0, 0],
                        [0, maxHeight - 1],
                        [maxWidth - 1, maxHeight - 1],
                        [maxWidth - 1, 0]])

    # Calculate the perspective transform matrix
    #print("Beofre try")

    

    try:

        #print(corners)
        #print(dest_pts)

        matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), dest_pts.astype(np.float32))
        #print("Passed matrix")

        corrected_img = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

        x, y, w, h = 0,0,maxWidth, maxHeight

        corrected_img = corrected_img[y:y + h, x:x + w]

        #filename = "result.png"
        #cv2.imwrite(filename, corrected_img) 

        returner = checkIfTall(corrected_img, corners)

        #filename = "QuickTest3.png"
        #cv2.imwrite(filename,  returner) 

        return returner
    

    except Exception:
        print("Err")

    return img

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


#This method will need to be fixed later, to see if something is actually upright
def checkIfTall(image, corners):
    # Read the image
        
    if image is None:
        print("Error: Image not loaded.")
        return None

    # Check if the image is wider than taller
    if image.shape[1] < image.shape[0]:
        #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image
    
    if image.shape[1] > image.shape[0]:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        #print("Rotate")
        return image


def checkAlignment():
    pass



def findCardScryfall(probableName):

    url = f"https://api.scryfall.com/cards/named?fuzzy={probableName}"

    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for any HTTP errors

        # Parse the JSON response
        data = response.json()

        #print("woah")

        # Extract the list of cards from the response
        

        return data

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None


    pass


def getSets(dataObject):

    #we will do url calling again

    url = dataObject['prints_search_uri']
    #print(url)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for any HTTP errors

        # Parse the JSON response
        sets = []
        data = response.json()
        for entry in data['data']:
            #print(entry['set'])
            sets.append(entry['set'])

       # Extract the list of cards from the response
        

        return sets[2:]
    
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None
    


def findSet(setList, card):

    mainList = "setList.csv"
    
    trueList = list(csv.reader(open(mainList)))

    gen = (y for x in trueList for y in x)


    #print(trueList[120])

   # print("LEA" in gen)

    for set in setList:
        if set in setList:
            pass
            #print("Found")


    #UNFINISHED
    
    #first step is to check against our local listing
    #for entry in setList:

