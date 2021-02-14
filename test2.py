from cv2 import cv2
from random import randrange
import time
import json
import numpy as np 


class Test:

    def __init__(self, result_path, img_path, img_list, numero_immagini):
        self.result_path = result_path
        self.img_path = img_path
        self.img_list = img_list
        self.numero_immagini = numero_immagini

    def features_detection(self, algoritmo):
        """
        Metodo che identifica le features di due immagini adiacenti, esegue l'algoritmo
        brute-force per trovare le corrispondenze ed infine tramite ransac determina
        il numero di inliers ed outliers dei match.
        
        algoritmo: algoritmo utilizzato per il test
        """

        match_array = []
        match_return = 0
        inliers_array = []
        inliers_return = 0
        outliers_array = []
        outliers_return = 0
        time_array = []
        time_return = 0
        total_time = 0

        data = {}
        data['risultato'] = []

        start_totale_time = time.time()

        for i in range(len(self.img_list) - 1):

            start_time = time.time()
            
            img1 = cv2.imread(self.img_path + '\\originale\\' + self.img_list[i], cv2.IMREAD_UNCHANGED)
            img2 = cv2.imread(self.img_path + '\\originale\\' + self.img_list[i + 1], cv2.IMREAD_UNCHANGED)
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            if algoritmo == 'SIFT':
                sift = cv2.SIFT_create(nfeatures=4000)
                kp_1, des_1 = sift.detectAndCompute(img1_gray, None)
                kp_2, des_2 = sift.detectAndCompute(img2_gray, None)

            elif algoritmo == 'ORB':
                orb = cv2.ORB_create(nfeatures=4000)
                kp_1, des_1 = orb.detectAndCompute(img1_gray, None)
                kp_2, des_2 = orb.detectAndCompute(img2_gray, None)

            elif algoritmo == 'AKAZE':
                akaze = cv2.AKAZE_create()
                kp_1, des_1 = akaze.detectAndCompute(img1_gray, None)
                kp_2, des_2 = akaze.detectAndCompute(img2_gray, None)
            
            elif algoritmo == 'BRISK':
                brisk = cv2.BRISK_create()
                kp_1, des_1 = brisk.detectAndCompute(img1_gray, None)
                kp_2, des_2 = brisk.detectAndCompute(img2_gray, None)
            
            else:
                print("Nessun algoritmo selezionato")

            #Brute-force match
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des_1, des_2)
            dmatches = sorted(matches, key = lambda x:x.distance)

            src_pts  = np.float32([kp_1[m.queryIdx].pt for m in dmatches]).reshape(-1,1,2)
            dst_pts  = np.float32([kp_2[m.trainIdx].pt for m in dmatches]).reshape(-1,1,2)

            #Identifico inliers ed outliers dei match
            if(len(matches) >= 4):
                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                inliers = 0
                outliers = 0
                for j in range(len(mask)):
                    if mask[j] == 1:
                        inliers = inliers + 1
                    else:
                        outliers = outliers + 1
            else:
                inliers = 0
                outliers = 0
            

            for j in range(len(mask)):
                    if mask[j] == 1:
                        inliers = inliers + 1
                    else:
                        outliers = outliers + 1

            match_array.append(len(matches))
            inliers_array.append(inliers)
            outliers_array.append(outliers)

            end_time = time.time()

            time_array.append(end_time - start_time)
        
        end_totale_time = time.time()

        total_time = end_totale_time - start_totale_time

        for i in range(0, self.numero_immagini - 1):
            match_return = match_return + match_array[i]
            inliers_return = inliers_return + inliers_array[i]
            outliers_return = outliers_return + outliers_array[i]
            time_return = time_return + time_array[i]

        data['risultato'].append({
            'match': match_return / (self.numero_immagini - 1),
            'inliers': inliers_return / (self.numero_immagini - 1),
            'outliers': outliers_return / (self.numero_immagini - 1),
            'time': time_return / (self.numero_immagini - 1),
            'total_time': total_time
        })
        
        with open(self.result_path + algoritmo + '\\risultato.txt', 'w') as outfile:
            json.dump(data, outfile)

if __name__ == '__main__':
    """
        Test per calcolare la robustezza degli algoritmi di Features Detection
        AKAZE, BRISK, ORB e SIFT per il processo di creazione del mosaico.
        Nella cartella 'Risultati2/nome_algoritmo/risultato.txt' possono essere
        reperiti i risultati ottenuti per quel determinato algoritmo
        
        img_path: percorso nel quale sono posizionate le immagini da analizzare
        img_list: lista contenente i nomi in ordine delle immagini, il ciclo for riempie la lista
        in questo modo ("0.tif", "1.tif", ..., "i.tif")
        algoritmo: algoritmo selezionato per l'analisi (AKAZE, BRISK, ORB, SIFT)
    """

    result_path = 'C:\\Users\\Carlo\\Workplace\\Python\\Test\\Risultati2\\'
    img_path = 'C:\\Users\\Carlo\\Workplace\\Python\\Test\Immagini\\'
    img_list = []
    algoritmo = 'AKAZE'
    numero_immagini = 5
    
    test = Test(result_path, img_path, img_list, numero_immagini)

    for i in range(0, numero_immagini):
        nome_immagine = str(i) + '.tif'
        img_list.append(nome_immagine)

    test.features_detection(algoritmo)

