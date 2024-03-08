import cv2
import numpy as np


scaling = 0.3
images = []
for i in range(1,11):
    k = cv2.imread("data/"+str(i)+".jpeg")
    k = cv2.resize(k,(int(k.shape[1]*scaling),int(k.shape[0]*scaling)))
    k=cv2.GaussianBlur(k,[3,3],2)
    images.append(k)
print("Images read successfully")
clayes = []
clh = cv2.createCLAHE()

for imag in images:
    hsv = cv2.cvtColor(imag,cv2.COLOR_BGR2HSV)
    low = np.array([95,0,0])
    up = np.array([100,254,184])
    thresh = cv2.inRange(hsv,low,up)
    lab = cv2.cvtColor(imag,cv2.COLOR_BGR2LAB)
    finimag = lab
    finimag[:,:,0] = clh.apply(finimag[:,:,0])
    finimag = cv2.cvtColor(finimag,cv2.COLOR_LAB2BGR)
    finimag = cv2.GaussianBlur(finimag,(3,3),4)
    ret,claheThresh = cv2.threshold(cv2.cvtColor(finimag,cv2.COLOR_BGR2GRAY),50,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("Original",imag)
    cv2.imshow("CLAHE",finimag)
    cv2.imshow("HSV Thresh",thresh)
    cv2.imshow("Thresh on CLAHE",claheThresh)
    cv2.imshow("CLAHE BW",cv2.cvtColor(finimag,cv2.COLOR_BGR2GRAY))
    cv2.waitKey()
# selectNum = 4
# imgPath = "/data/"+str(selectNum)+".jpeg"
# hsvImg = cv2.cvtColor(images[4],cv2.COLOR_BGR2HSV)
# bwImg = cv2.cvtColor(images[4],cv2.COLOR_BGR2GRAY)
# ret,threshImg = cv2.threshold(bwImg,109,255,cv2.THRESH_BINARY )
# adapThresh = cv2.adaptiveThreshold(bwImg,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,2)
# threhImages = []
# hsvThreshes = []
# for image in images:
#     bww = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#     aThr = cv2.adaptiveThreshold(bww,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,13,2)
#     threhImages.append(aThr)
# low = np.array([95,0,0])
# up = np.array([100,254,184])
# # for image in threhImages:
# #     cv2.imshow("HEHE",image)
# #     cv2.waitKey(1000)
# # cv2.imshow("Dobbi",threhImages[0])
# # cv2.imshow("Dobb1i",threhImages[1])
# # cv2.imshow("Dob1bi",threhImages[2])
# # cv2.imshow("Do2bbi",threhImages[3])
# # cv2.imshow("Do3bbi",threhImages[4])
# # cv2.imshow("Do4bbi",threhImages[5])
# # cv2.imshow("Do5bbi",threhImages[6])
# # cv2.imshow("Do6bbi",threhImages[7])
# # cv2.imshow("Dob77bi",threhImages[8])
# # cv2.imshow("Dob22bi",threhImages[9])
# for imag in images:
#     hsvImg = cv2.cvtColor(imag,cv2.COLOR_BGR2HSV)
#     thresh = cv2.inRange(hsvImg,low,up)
#     hsvThreshes.append(thresh)

# for imag in hsvThreshes:
#     cv2.imshow("ggi",imag)
#     cv2.waitKey(1000)
    # bgr = imag
    # bgray = bgr[:,:,0]
    # ggray = bgr[:,:,1]
    # rgray = bgr[:,:,2]
    # lgray = lab[:,:,0]
    # agray = lab[:,:,1]
    # bbgray = lab[:,:,2]
    # hgray = hsv[:,:,0]
    # sgray = hsv[:,:,1]
    # vgray = hsv[:,:,2]
    # bcla = clh.apply(bgray)
    # gcla = clh.apply(ggray)
    # rcla = clh.apply(rgray)
    # lcla = clh.apply(lgray)
    # acla = clh.apply(agray)
    # bbcla = clh.apply(bbgray)
    # hcla = clh.apply(hgray)
    # scla = clh.apply(sgray)
    # vcla = clh.apply(vgray)

    # finbgr = np.zeros(imag.shape)
    # finhsv = np.zeros(imag.shape)
    # finlab = np.zeros(imag.shape)

    # finbgr[:,:,1] =gcla//75
    # finbgr[:,:,2] =rcla//50
    # finbgr[:,:,0] =bcla//30


    # finlab[:,:,0] =lcla
    # finlab[:,:,1] =acla
    # finlab[:,:,2] =bbcla
    # # finlab = cv2.cvtColor(finlab,cv2.COLOR_LAB2BGR)

    # finhsv[:,:,0] =hcla
    # finhsv[:,:,1] =scla
    # finhsv[:,:,2] =vcla
    # # finhsv = cv2.cvtColor(finhsv,cv2.COLOR_HSV2BGR)

    # finimag = finbgr + finhsv + finlab
    # finimag//=3

    # imag = clh.apply(imag)
    # clayes.append(imag)
# for imag in clayes:
#     cv2.imshow("ggi",imag)
#     cv2.waitKey(1000)



