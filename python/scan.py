# Használat
# python scan.py --image page.jpg -o kesz.jpg

# modulok importálása, amiket használok
from skimage.filters import threshold_local
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import argparse
import cv2
import imutils
import os.path

def order_points(pts):
    # koordinátákból listát készítek,
    # úgy, hogy az első elem a listában a bal felső,
    # a második a jobb felső, a harmadik pedig a jobb alsó,
    # a negyedik pedig az bal alsó kordinátákat tartalmazza.
    rect = np.zeros((4, 2), dtype = "float32")

    # ball fölső pont - legkissebb szumma
    # ball alsó pont - legnagyobb szumma
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # különbségszámólás
    # jobb fölső legkissebb diferencia
    # jobb allsó legnagyobb különbség
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Viszaadom a számolt kordinátákat
    return rect

def four_point_transform(image, pts):
    # Berendezem a kapott kordinátákat, majd külön változókba szétbontom.
    # top left , top right... értelemszerűen.
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # itt kiszámolom a kép szélességét,
    # a legnagyobb különbség a ball és jobb oldal között.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # itt kiszámolom a kép magasságát, ami a legnagyobb távolság lesz
    # a jobb felső és a jobb alsó, illetve a másik oldalon
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # most átalakítjuk a képet, a kiszámolt méretekre
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
	[0, 0],
	[maxWidth - 1, 0],
	[maxWidth - 1, maxHeight - 1],
	[0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped
def invertal(image):
    image_i = (255-image)
    #image_i = cv2.bitwise_not(image)
    return image_i

# Itt vesszem át a paramétereket
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
    help = "A scannelendő kép elérési útja")
ap.add_argument("-o", "--output", required = True,
    help = "A kimeneti kép elérési útvonala")
# opcionális Paraméterek

#ORIGINAL KÉP---------------------------------------
ap.add_argument("-m_o", "--median_o", required = False,
    help = "median filterezés-Só és bors zaj-ellen")
#Zajszűrés
ap.add_argument("-g_o", "--gauss_o", required = False,
    help = "Gaussian zajcsökkentés-holmályosítja a képet")
#Élesítés - A zajokat is kiemeli
ap.add_argument("-b_o", "--bilinear_o", required = False,
    help = "bilineáris filter")
#blur = cv2.bilateralFilter(img,9,75,75) textúrákat csökkenti, az éleket meghagyja
ap.add_argument("-n_o", "--invert_o", required = False,
    help = "negatív kép")
# függvény-ok
ap.add_argument("-c_o", "--clahe_o", required = False,
    help = "kotraszt normalizálás")
#Kvantálás
ap.add_argument("-kv_o", "--kvantalas_o", required = False,type=int,
    help = "Kvantálás-quantization")


#VÁGOTT KÉP-----------------------------------------
ap.add_argument("-m", "--median", required = False,
    help = "median filterezés-Só és bors zaj-ellen")
#Zajszűrés
ap.add_argument("-g", "--gauss", required = False,
    help = "Gaussian zajcsökkentés-holmályosítja a képet")
#Élesítés - A zajokat is kiemeli
ap.add_argument("-b", "--bilinear", required = False,
    help = "bilineáris filter")
#blur = cv2.bilateralFilter(img,9,75,75) textúrákat csökkenti, az éleket meghagyja
ap.add_argument("-n", "--invert", required = False,
    help = "negatív kép")
# függvény-ok
ap.add_argument("-c", "--clahe", required = False,
    help = "kotraszt normalizálás")
#Kvantálás
ap.add_argument("-kv", "--kvantalas", required = False,type=int,
    help = "Kvantálás-quantization")


args = vars(ap.parse_args())

# Átveszem a paramétereket
imgpath= args["image"]
outpath = args["output"]

# A kép beolvasása, ha létezik.
if os.path.isfile(imgpath):
    # leméretezem a képet, hogy könnyebb legyen vele dolgozni
    image = cv2.imread(imgpath)
    if (args["median_o"]):
        print("STEP 4: Median Filterezés;")
        median = cv2.medianBlur(image,5) # - 50%
        #cv2.imwrite(outpath+"median.jpeg", median)
        image = median
    # Zajszűrés
    if (args["gauss_o"]):
        print("STEP 4: Gaussian zajcsökkentés;")
        # params - kernel méret
        glauss= cv2.GaussianBlur(image, (5, 5), 0)
#        cv2.imwrite(outpath + "gauss.jpeg", glauss)
        image = glauss
    #	help="Gaussian zajcsökkentés-holmályosítja a képet")

    # Élesítés - A zajokat is kiemeli
    if (args["bilinear_o"]):
        print("STEP 4: Bilinearis Filterezés;")
        bilinear = cv2.bilateralFilter(image,9,75,75)
#        cv2.imwrite(outpath + "bilinear.jpeg", bilinear)
        image = bilinear
    #	textúrákat csökkenti, az éleket meghagyja

    if (args["invert_o"]):
        print("STEP 4: Invertálás;")
        invert = invertal(image)
        #cv2.imwrite(outpath + "invert.jpeg", invert)
        image = invert
    #	help="negatív kép")
    if (args["clahe_o"]):
        print("STEP 4: CLAHE;")
        bgr = image
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#        cv2.imwrite(outpath + "clahe.jpeg", bgr)
        image = bgr

    if (args["kvantalas_o"]):
        print("STEP 4: Kvantalas;")
        (h, w) = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = MiniBatchKMeans(n_clusters=args["kvantalas_o"])
        labels = clt.fit_predict(image)
        quant = clt.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((h, w, 3))
        image = image.reshape((h, w, 3))
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        #image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        image = quant
        #cv2.imwrite(outpath + "kvantalas.jpeg", quant)
    # 	help="Kvantálás-quantization")

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height = 500)


    # Átalakítom szűrkeskálásra
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # esetekben a GaussianBlur javíthatja a scannelés sikerességét,
    # mivel elmossa a képet így a nem releváns élek kikerülnek belőle
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # erősen gondolkodok rajta, hogy külön paraméterből kellene jönnie

    # Canny kép detektálás, hiszterézissel
    edged = cv2.Canny(gray, 75, 200)

    print('STEP 1: Él keresés;')
    #cv2.imshow("Image", image)
    #cv2.imshow("Edged", edged)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Kotnúrok keresése
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]


    for c in cnts:
        # approximate the contour
       peri = cv2.arcLength(c, True)
       approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # ha a talált kontúrnak 4 pontja van akkor,feltételezzük, hogy megtaláltuk a dokumentumot
       if len(approx) == 4:
           screenCnt = approx
           break

    # kontúr mutatása
    print("STEP 2: Kontúrok keresése;")
    try:
       cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    except NameError:
        print ("Nincs elkülöníthető kontúr!")
        exit()
    cv2.imwrite(outpath+".jpg", image)
    print("STEP 3: Transzformáció;")
    #transzformálás
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    #cv2.imshow("warped", warped)
    cv2.imwrite(outpath+".jpeg", warped)
    if (args["median"]):
        print("STEP 4: Median Filterezés;")
        median = cv2.medianBlur(warped,5) # - 50%
        cv2.imwrite(outpath+"median.jpeg", median)
#        warped = median
    # Zajszűrés
    if (args["gauss"]):
        print("STEP 4: Gaussian zajcsökkentés;")
        # params - kernel méret
        glauss= cv2.GaussianBlur(warped, (5, 5), 0)
        cv2.imwrite(outpath + "gauss.jpeg", glauss)
#        warped = glauss
    #	help="Gaussian zajcsökkentés-holmályosítja a képet")

    # Élesítés - A zajokat is kiemeli
    if (args["bilinear"]):
        print("STEP 4: Bilinearis Filterezés;")
        bilinear = cv2.bilateralFilter(warped,9,75,75)
        cv2.imwrite(outpath + "bilinear.jpeg", bilinear)
#        warped = bilinear
    #	textúrákat csökkenti, az éleket meghagyja

    if (args["invert"]):
        print("STEP 4: Invertálás;")
        invert = invertal(warped)
        cv2.imwrite(outpath + "invert.jpeg", invert)
#        warped = invert
    #	help="negatív kép")
    if (args["clahe"]):
        print("STEP 4: CLAHE;")
        bgr = warped
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        cv2.imwrite(outpath + "clahe.jpeg", bgr)
#        warped = bgr

    if (args["kvantalas"]):
        print("STEP 4: Kvantalas;")
        (h, w) = warped.shape[:2]
        image = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
        image = image.reshape((image.shape[0] * image.shape[1], 3))
        clt = MiniBatchKMeans(n_clusters=args["kvantalas"])
        labels = clt.fit_predict(image)
        quant = clt.cluster_centers_.astype("uint8")[labels]
        quant = quant.reshape((h, w, 3))
        image = image.reshape((h, w, 3))
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        #warped = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        cv2.imwrite(outpath + "kvantalas.jpeg", quant)
    # 	help="Kvantálás-quantization")

    print ("DONE!")
else:
    print("File nem létezik!")
