import cv2
import numpy as np
import library
import time
import lib2
import multiprocessing
import ocrTest


class card:
    def __init__(self):
        
        self.Image = ""
        self.name = ""



class mtgScanner:

    def __init__(self):
        #I have some weird numbers set, but works in my very restricted env
        self.cannyDetector = cv2.cuda.createCannyEdgeDetector(13, 170)
        self.gpuFrame = cv2.cuda_GpuMat()
        self.gaussFilter = cv2.cuda.createGaussianFilter(self.gpuFrame.type(), -1, (5, 5), 0)
        self.edges = ""

        ##CHANGE THIS TO RUN NORMAL
        self.gpu = False
        self.frameNormal = None

        #Allows asyncronous calls to gpu for data work
        self.stream = cv2.cuda_Stream()

        #optional options here just because
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.frameRate = 5
        
        self.resolutionX = 1920
        self.resolutionY = 1080
        self.cap.set(3, self.resolutionX)
        self.cap.set(4, self.resolutionY)

        #Inspired by https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector/tree/master

        self.ratio1 = self.resolutionX / 640
        self.ratio2 = self.resolutionY / 480


        #Interesting area, helps us find the image as a mtg card or not
        #default number is for default cam size
        #default 640 x 480, found default numbers to help sizing
        self.totalRatio = self.ratio1 + self.ratio2

        self.totalRatio = 4.5



        #Override because the math just isnt working right now, more research needed
        self.cardMax = (120000 * self.totalRatio) 
        self.cardMin = (25000 * self.totalRatio) 

            



    def processFrame(self):
        self.gpuFrame = cv2.cuda.cvtColor(self.gpuFrame, cv2.COLOR_BGR2GRAY, stream=self.stream)
        self.gpuFrame = self.gaussFilter.apply(self.gpuFrame)
        gpuEdge = self.cannyDetector.detect(self.gpuFrame)

            #edges processing for some contours in the image
        self.edges = gpuEdge.download(stream=self.stream)

        contours, heirarchy = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, heirarchy
    
    def processFrameNonGPU(self):
        
        gray = cv2.cvtColor(self.frameNormal, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        #thresholding using OTSU and Binary, its really neat
        _, th3 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        #Artbitray values considinger OTSU does its own adaptation, but I couldnt get it to work otherwise?
        threshold1 = 50
        threshold2 = 150

        edges = cv2.Canny(blurred, threshold1, threshold2)
        contours, heirarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours, heirarchy
          


    
    def contourSort(self, contours, heirarchy):
        index_sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)
        cnts_sort = []
        hier_sort = []
        listCount = []
        cnt_is_card = np.zeros(len(contours),dtype=int)

        for i in index_sort:
            cnts_sort.append(contours[i])
            hier_sort.append(heirarchy[0][i])

        for i in range(len(cnts_sort)):
            size = cv2.contourArea(cnts_sort[i])
            peri = cv2.arcLength(cnts_sort[i],True)
            approx = cv2.approxPolyDP(cnts_sort[i],0.01*peri,True)
            
            if ((size < self.cardMax) and (size > self.cardMin)
                and (hier_sort[i][3] == -1) and (len(approx) == 4)):
                #print("DING DING DING")
                cnt_is_card[i] = 1
                listCount.append(cnts_sort[i])

        return listCount



        
def main():

    Scanner = mtgScanner()
    print("Intiializing")
    cv2.namedWindow('Live View with Bounding Boxes')
    #cv2.namedWindow('ROI')  
    #cv2.namedWindow('output')

    while True:
        prev = 0

        time_elapsed = time.time() - prev
        res, image = Scanner.cap.read()

        if time_elapsed > 1./Scanner.frameRate:
            prev = time.time()
            #Capture frame from the webcam
            _, frame = Scanner.cap.read()           

            if Scanner.gpu==True:
                Scanner.gpuFrame.upload(frame)
                contours, heirarchy = Scanner.processFrame()
            else:
                #Normal non Cuda compilation
                print("Non GPU")
                Scanner.frameNormal = frame
                contours, heirarchy = Scanner.processFrameNonGPU()

            if (contours == 0):
                print("Nothing found")
                continue
 
            listCount = Scanner.contourSort(contours, heirarchy)
            print(listCount)


            for idx, contour in enumerate(listCount):
                x, y, w, h = cv2.boundingRect(contour)

                #frame = real


                #BOX
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                approx = cv2.approxPolyDP(contour, 0.009 * cv2.arcLength(contour, True), True) 
                cv2.drawContours(frame, [approx], 0, (0, 0, 255), 5)

                #get the roi, the region of interest, and split it out so we know what we lookin at
                roi = frame[y:y + h, x:x + w]
                #cornersBack = roi
                cornersBack = library.alignCard(roi, approx)

                for corner in cornersBack:
                    cv2.circle(frame, tuple(corner), 10, (0, 255, 0), -1)
                
                #cv2.imshow('ROI', roi)            

                correctedIM = lib2.correct_card_orientation(frame, cornersBack, contour)
                fullCrop = correctedIM

                correctedIM = library.crop_top_percentage(correctedIM)

                filename = "FinalTest.png"
                cv2.imwrite(filename,  correctedIM) 

                # Denoise?
                #correctedIM = cv2.fastNlMeansDenoisingColored(correctedIM, 3, 3, 7, 51)

                #kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
                correctedIM = cv2.filter2D(correctedIM, -1, kernel)

                #Denoising experiments
                #correctedIM = cv2.cvtColor(correctedIM, cv2.COLOR_BGR2GRAY)
                #correctedIM = cv2.fastNlMeansDenoising(correctedIM, 10, 10, 7, 21)


                # Deep network attempt to upscale the image, just curious                
                #sr = cv2.dnn_superres.DnnSuperResImpl_create()
                #path = "EDSR_x4.pb"
                #sr.readModel(path)
                #sr.setModel("edsr", 2)
                #correctedIM = sr.upsample(correctedIM)

                filename = "QuickTest.png"
                cv2.imwrite(filename,  correctedIM) 

                
                #cv2.imshow('output', correctedIM)

                imageWrite, extra = ocrTest.ocrTest(correctedIM)

                extractedText = ""
                for (bbox, entry, prob) in extra:
                    extractedText = extractedText + " " + entry


                font_scale = 2
                font_thickness = 2
                text_size = cv2.getTextSize(extractedText, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x = x + (w - text_size[0]) // 2
                text_y = y + h + text_size[1] + 5

                cardRet = lib2.findCardScryfall(extractedText)
                if (cardRet != None):               #print(cardRet)

                    #This section used only to display Name
                    print(cardRet['name'])

     

                    cv2.putText(frame, cardRet['name'], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (45, 255, 45), font_thickness, cv2.LINE_AA)

                    #setRet = lib2.getSets(cardRet)
                    #mvresult = lib2.findSet(setRet, roi)
                    #print("===============")
                    #print(setRet)
                    #print("===============")


                #Set tester
                set = library.crop_set(fullCrop)
                #filename = "SetTest.png"
                #cv2.imwrite(filename,  set) 

                                
                GuessedSet = library.checkSet(set, "2017")
                
                nameOBj = ""
                if (cardRet == None):
                    nameOBj = "Unknown"
                else:
                    nameOBj = cardRet['name']

                fullString = nameOBj + " Set: " + GuessedSet

                cv2.putText(frame, fullString, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (45, 255, 45), font_thickness, cv2.LINE_AA)    


                #cv2.imshow('output', imageWrite)

               
        #Numnber of cards
        cv2.putText(frame, f"Cards Detected: {len(listCount)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        
        cv2.imshow('Live View with Bounding Boxes', frame)
        #cv2.imshow('Canny Edges', Scanner.edges)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release the VideoCapture and close all windows
    Scanner.cap.release()
    cv2.destroyAllWindows()




            



            













        


if __name__ == '__main__':
    main()
        