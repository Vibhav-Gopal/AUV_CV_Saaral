import cv2
import numpy as np

def gradient(src):
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(src, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(src, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = np.expand_dims(cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0), axis=2)
    return grad
scaling = 0.35
images = []
for i in range(20,35):
    k = cv2.imread("data/"+str(i)+".jpg")
    k = cv2.resize(k,(int(k.shape[1]*scaling),int(k.shape[0]*scaling)))
    k=cv2.GaussianBlur(k,[7,7],0)
    images.append(k)
print("Images read successfully")
clayes = []
clh = cv2.createCLAHE()

for imag in images:
    hsv = cv2.cvtColor(imag,cv2.COLOR_BGR2HSV)
    bw = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)
    low = np.array([95,0,0])
    up = np.array([100,254,184])
    thresh = cv2.inRange(hsv,low,up)
    lab = cv2.cvtColor(imag,cv2.COLOR_BGR2LAB)
    finimag = lab
    finimag[:,:,0] = clh.apply(finimag[:,:,0])
    finimag = cv2.cvtColor(finimag,cv2.COLOR_LAB2BGR)
    finimag = cv2.GaussianBlur(finimag,(5,5),4)
    ret,claheThresh = cv2.threshold(cv2.cvtColor(finimag,cv2.COLOR_BGR2GRAY),50,255,cv2.THRESH_BINARY_INV)
    cv2.imshow("Original",imag)
    cv2.imshow("CLAHE",finimag)
    cv2.imshow("Thresh on CLAHE",claheThresh)
    # cv2.imshow("CLAHE BW",cv2.cvtColor(finimag,cv2.COLOR_BGR2GRAY))
    ret, sobelx = cv2.threshold(cv2.Sobel(hsv[:,:,2],0,dx=1,dy=0) , 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ret, sobely = cv2.threshold(cv2.Sobel(hsv[:,:,2],0,dx=0,dy=1),0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    sobelFin = sobelx + sobely
    roshGrad = gradient(bw)
    ret, roshGrad = cv2.threshold(roshGrad,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #Paper thresh
    rchannel = imag[:,:,2]
    bchannel = imag[:,:,0]
    gchannel = imag[:,:,1]
    rchannel = clh.apply(rchannel)
    bchannel = clh.apply(bchannel)
    gchannel = clh.apply(gchannel)
    rchannel = cv2.GaussianBlur(rchannel,[5,5],2)
    bchannel = cv2.GaussianBlur(bchannel,[5,5],2)
    gchannel = cv2.GaussianBlur(gchannel,[5,5],2)
    clhrgb = np.zeros_like(imag)
    clhrgb[:,:,0] = bchannel
    clhrgb[:,:,1] = gchannel
    clhrgb[:,:,2 ] = rchannel
    clhrgb = cv2.GaussianBlur(clhrgb,[3,3],9)
    clhrgbhsv = cv2.cvtColor(clhrgb,cv2.COLOR_BGR2HSV)
    lower = np.array([0,110,0])
    upper = np.array([60,255,255])
    threshNewCl = cv2.inRange(clhrgbhsv,lower,upper)
    kernel = np.ones((5, 5), np.uint8) 
    threshNewClero = cv2.erode(threshNewCl,kernel,iterations=2)
    threshNeweroDila = cv2.dilate(threshNewClero,kernel,iterations=8)
    #masking
    masked = np.zeros_like(clhrgb)

    masked[:,:,0] = cv2.bitwise_and(threshNeweroDila,clhrgb[:,:,0])
    masked[:,:,1] = cv2.bitwise_and(threshNeweroDila,clhrgb[:,:,1])
    masked[:,:,2] = cv2.bitwise_and(threshNeweroDila,clhrgb[:,:,2 ])
    cv2.imshow("werid",clhrgb)
    cv2.imshow("werid hsv thresh",threshNewCl)
    cv2.imshow("werid hsv thresh eroded",threshNewClero)
    cv2.imshow("werid hsv thresh eroded n dilated",threshNeweroDila)
    cv2.imshow("werid hsv thresh eroded n dilated and segmented",masked)


    # cv2.imshow("RoshGrad",roshGrad)
    # cv2.imshow("Sobel X enhanced",sobelx)
    # cv2.imshow("Sobel fin",sobelFin )
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



