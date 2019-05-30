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
# Függvénydefiníciók
def order_points(pts):
    # koordinátákból listát készítek,
    # úgy, hogy az első elem a listában a bal felső,
    # a második a jobb felső, a harmadik pedig a jobb alsó,
    # a negyedik pedig az bal alsó kordinátákat tartalmazza.
    rect = np.zeros((4, 2), dtype="float32")

    # ball fölső pont - legkissebb szumma
    # ball alsó pont - legnagyobb szumma
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # különbségszámólás
    # jobb fölső legkissebb diferencia
    # jobb allsó legnagyobb különbség
    diff = np.diff(pts, axis=1)
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
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped
def median_filter(i, perc=5,output=False):
    print("STEP 4: Median Filterezés;")
    median = cv2.medianBlur(i, perc)
    if (show):
        cv2.imshow("Median", median)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if output:
        cv2.imwrite(outpath + "median.jpeg", median)
    return median
def gauss_filter(i,kernel=5,output=False):
    print("STEP 4: Gaussian zajcsökkentés;")
    # params - kernel méret
    gauss = cv2.GaussianBlur(i, (kernel, kernel), 0)
    if (show):
        cv2.imshow("gauss", gauss)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if output:
            cv2.imwrite(outpath + "gauss.jpeg", gauss)
    return gauss
def bilinear_filter(i,output=False):
    print("STEP 4: Bilinearis Filterezés;")
    bilinear = cv2.bilateralFilter(i,9,75,75)
    if (show):
        cv2.imshow("bilinear", bilinear)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if output:
        cv2.imwrite(outpath + "bilinear.jpeg", bilinear)
    return bilinear
def invert_filter(i,output=False):
    image_i = (255 - i)
    # image_i = cv2.bitwise_not(image)
    if (show):
        cv2.imshow("invert", image_i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if output:
        cv2.imwrite(outpath + "invert.jpeg", image_i)
    return image_i
def clahe_filter(i,output=False):
    print("STEP 4: CLAHE;")
    bgr = i
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if (show):
        cv2.imshow("clahe", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if output:
        cv2.imwrite(outpath + "clahe.jpeg", bgr)
    return bgr
def kvantalo_filter(i,output=False):
    print("STEP 4: Kvantalas;")
    (h, w) = i.shape[:2]
    image = cv2.cvtColor(i, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = MiniBatchKMeans(n_clusters=args["kvantalas_o"])
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    # image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    if (show):
        cv2.imshow("quant ", quant)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if output:
        cv2.imwrite(outpath + "kvantalas.jpeg", quant)
    return quant

# Így vesszem át a paramétereket
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="A scannelendő kép elérési útja")
ap.add_argument("-o", "--output", required=True,
                help="A kimeneti kép elérési útvonala")
# opcionális Paraméterek
# ORIGINAL KÉP---------------------------------------
ap.add_argument("-m_o", "--median_o", required=False,
                help="median filterezés-Só és bors zaj-ellen")
# Zajszűrés
ap.add_argument("-g_o", "--gauss_o", required=False,
                help="Gaussian zajcsökkentés-holmályosítja a képet")
# Élesítés - A zajokat is kiemeli
ap.add_argument("-b_o", "--bilinear_o", required=False,
                help="bilineáris filter")
# blur = cv2.bilateralFilter(img,9,75,75) textúrákat csökkenti, az éleket meghagyja
ap.add_argument("-n_o", "--invert_o", required=False,
                help="negatív kép")
# függvény-ok
ap.add_argument("-c_o", "--clahe_o", required=False,
                help="kotraszt normalizálás")
# Kvantálás
ap.add_argument("-kv_o", "--kvantalas_o", required=False, type=int,
                help="Kvantálás-quantization")
# VÁGOTT KÉP-----------------------------------------
ap.add_argument("-m", "--median", required=False,
                help="median filterezés-Só és bors zaj-ellen")
# Zajszűrés
ap.add_argument("-g", "--gauss", required=False,
                help="Gaussian zajcsökkentés-holmályosítja a képet")
# Élesítés - A zajokat is kiemeli
ap.add_argument("-b", "--bilinear", required=False,
                help="bilineáris filter")
# blur = cv2.bilateralFilter(img,9,75,75) textúrákat csökkenti, az éleket meghagyja
ap.add_argument("-n", "--invert", required=False,
                help="negatív kép")
# függvény-ok
ap.add_argument("-c", "--clahe", required=False,
                help="kotraszt normalizálás")
# Kvantálás
ap.add_argument("-kv", "--kvantalas", required=False, type=int,
                help="Kvantálás-quantization")
# TERMINÁLBÓL / WINDOWS KONZOLBÓL VALÓ FUTTATÁSHOZ
ap.add_argument("-t", "--terminal", required=False, type=int,
                help="Bekapcsolja az opencv vizualizációt")
args = vars(ap.parse_args())

# Átveszem a paramétereket
imgpath = args["image"]
outpath = args["output"]
if args["terminal"]!=None:
    show = True
else:
    show = False
if os.path.isfile(imgpath):
    image = cv2.imread(imgpath)
    # leméretezem a képet, hogy könnyebb legyen vele dolgozni
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # pre-módosítások
    if args["median_o"] != None:
        image=median_filter(image,args["median_o"])
    if args["gauss_o"] != None:
        image=gauss_filter(image,args["gauss_o"])
    if args["bilinear_o"] != None:
        image=bilinear_filter(image)
    if args["invert_o"] != None:
        image=invert_filter(image)
    if args["clahe_o"] != None:
        image=clahe_filter(image)
    if args["kvantalas_o"] != None:
        image=kvantalo_filter(image)

    # Átalakítom szűrkeskálásra
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny kép detektálás, hiszterézissel
    edged = cv2.Canny(gray, 75, 200)

    print('STEP 1: Él keresés;')
    cv2.imshow("Image", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Kotnúrok keresése
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

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
        print("Nincs elkülöníthető kontúr!")
        exit()
    cv2.imwrite(outpath + ".jpg", image)
    print("STEP 3: Transzformáció;")
    # transzformálás
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
    # cv2.imshow("warped", warped)
    cv2.imwrite(outpath + ".jpeg", warped)
    if args["median_o"] != None:
        warped=median_filter(warped,args["median_o"],True)
    if args["gauss_o"] != None:
        warped=gauss_filter(warped,args["gauss_o"],True)
    if args["bilinear_o"] != None:
        warped=bilinear_filter(warped,True)
    if args["invert_o"] != None:
        warped=invert_filter(warped,True)
    if args["clahe_o"] != None:
        warped=clahe_filter(warped,True)
    if args["kvantalas_o"] != None:
        warped=kvantalo_filter(warped,True)
    cv2.imwrite(outpath + "warped.jpg", warped)

else:
    print("File nem létezik")
