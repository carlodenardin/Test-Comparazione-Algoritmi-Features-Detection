from cv2 import cv2
from random import randrange
import os
import time
import json

class Test:

    def __init__(self, result_path, img_path, img_list, numero_immagini):
        self.result_path = result_path
        self.img_path = img_path
        self.img_list = img_list
        self.numero_immagini = numero_immagini

    def resize_rotate_image(self, img_name, scale):
        """
            Metodo che data un'immagine e una lista di scale, scala l'immagine per tutti i valori all'interno della
            lista e ruota in maniera non deteriministica (90, 180, 270).

            img_name: immagine da modificare
            scale: scale da applicare all'immagine
        """
        

        img_open = cv2.imread(self.img_path + "\\originale\\" + img_name, cv2.IMREAD_UNCHANGED)

        for i in range(len(scale)):

            #Ridimensionamento
            width = int(img_open.shape[1] * scale[i] / 100)
            height = int(img_open.shape[0] * scale[i] / 100)
            dim = (width, height)

            resized_image = cv2.resize(img_open, dim, interpolation = cv2.INTER_AREA)
            
            #Rotazione
            random_rotation = randrange(3)

            if random_rotation == 0:
                resized_rotated_image = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
            elif random_rotation == 1:
                resized_rotated_image = cv2.rotate(resized_image, cv2.ROTATE_180)
            else:
                resized_rotated_image = cv2.rotate(resized_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

            #Salvo l'immagine scalata nella cartella corrispondente al suo ridimensionamento
            save_path = self.img_path + "\\" + str(scale[i]) + "\\" + img_name

            cv2.imwrite(save_path, resized_rotated_image)

    def features_detection(self, nome_immagine, scale, algoritmo):
        """
            Metodo che esegue l'estrapolazione delle features delle immagini originali e le
            corrispondenti immagini scalate. Inoltre, successivamente per ogni immagine originale
            e le immagini scalate esegue il match calcolando il numero medio di match identificati

        """

        #-- PARAMETRI DA CALCOLARE --#
        features_immagine_originale = 0
        features_immagine_scalata_array = []
        features_immagine_scalata = 0
        features_matched_array = []
        features_matched = 0
        tempo_features_detection_originale = 0
        tempo_features_detection_scalata_array = []
        tempo_features_detection_scalata = 0
        tempo_matching_array = []
        tempo_matching = 0

        #-- FEATURE DETECTION IMMAGINI ORIGINALI --#
        img_originale = cv2.imread(self.img_path + '\\originale\\' + nome_immagine, cv2.IMREAD_UNCHANGED)
        img_originale_grigia = cv2.cvtColor(img_originale, cv2.COLOR_BGR2GRAY)

        start_time = time.time()

        if algoritmo == 'SIFT':
            original_sift = cv2.SIFT_create(nfeatures=4000)
            original_kp, original_des = original_sift.detectAndCompute(img_originale_grigia, None)

        elif algoritmo == 'ORB':
            original_orb = cv2.ORB_create(nfeatures=4000)
            original_kp, original_des = original_orb.detectAndCompute(img_originale_grigia, None)

        elif algoritmo == 'AKAZE':
            original_akaze = cv2.AKAZE_create()
            original_kp, original_des = original_akaze.detectAndCompute(img_originale_grigia, None)
        
        elif algoritmo == 'BRISK':
            original_brisk = cv2.BRISK_create()
            original_kp, original_des = original_brisk.detectAndCompute(img_originale_grigia, None)
        
        else:
            print("Nessun algoritmo selezionato")

        end_time = time.time()
        tempo_features_detection_originale = end_time - start_time

        #-- FEATURE DETECTION IMMAGINI SCALATE --#

        for i in range(len(scale)):
            img_scalata = cv2.imread(self.img_path + '\\' + str(scale[i]) + '\\' + nome_immagine, cv2.IMREAD_UNCHANGED)
            print(self.img_path + '\\' + str(scale[i]) + '\\' + nome_immagine)
            img_scalata_grigia = cv2.cvtColor(img_scalata, cv2.COLOR_BGR2GRAY)

            start_time = time.time()

            if algoritmo == 'SIFT':
                scaled_sift = cv2.SIFT_create(nfeatures=4000)
                scaled_kp, scaled_des = scaled_sift.detectAndCompute(img_scalata_grigia, None)

            elif algoritmo == 'ORB':
                scaled_orb = cv2.ORB_create(nfeatures=4000)
                scaled_kp, scaled_des = scaled_orb.detectAndCompute(img_scalata_grigia, None)

            elif algoritmo == 'AKAZE':
                scaled_akaze = cv2.AKAZE_create()
                scaled_kp, scaled_des = scaled_akaze.detectAndCompute(img_scalata_grigia, None)
            
            elif algoritmo == 'BRISK':
                scaled_brisk = cv2.BRISK_create()
                scaled_kp, scaled_des = scaled_brisk.detectAndCompute(img_scalata_grigia, None)
            
            else:
                print("Nessun algoritmo selezionato")
            
            end_time = time.time()

            tempo_features_detection_scalata_array.append(end_time - start_time)
            
            features_immagine_scalata_array.append(len(scaled_kp))
            
            divisore = 7

            start_time = time.time()

            if len(scaled_kp) > 1:

                #-- MATCH ORIGINALE E IMMAGINE SCALATA --#
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(original_des, scaled_des, k = 2)

                good = []

                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good.append([m])
                
                features_matched_array.append(len(good))

            else:

                features_matched_array.append(0)
                divisore = divisore - 1
        
            end_time = time.time()

            tempo_matching_array.append(end_time - start_time)

        features_immagine_originale = len(original_kp)
        
        for x in range(7):
            features_immagine_scalata = features_immagine_scalata + features_immagine_scalata_array[x]
            features_matched = features_matched + features_matched_array[x]
            tempo_features_detection_scalata = tempo_features_detection_scalata + tempo_features_detection_scalata_array[x]
            tempo_matching = tempo_matching + tempo_matching_array[x]

        return features_immagine_originale, (features_immagine_scalata / 7), (features_matched / divisore), tempo_features_detection_originale, (tempo_features_detection_scalata / 7), (tempo_matching / 7), features_immagine_scalata_array[0], features_matched_array[0], features_immagine_scalata_array[1], features_matched_array[1], features_immagine_scalata_array[2], features_matched_array[2], features_immagine_scalata_array[3], features_matched_array[3], features_immagine_scalata_array[4], features_matched_array[4], features_immagine_scalata_array[5], features_matched_array[5], features_immagine_scalata_array[6], features_matched_array[6]
        

    def run(self, scale, algoritmo):
        """
            Metodo principale il quale gestisce l'estrapolazione dei risultato. Per ogni immagine
            all'interno della cartella 'Immagini/originale/' chiama il metodo features detection.

            scale: scale da analizzare della stessa immagine
            algoritmo: algoritmo utilizzato per il features detection

            Infine il metodo principale salve i risultati nella cartella 'Risultati1/algoritmo/risultati.txt'
            nella quale algoritmo corrisponde al nome dell'algoritmo utilizzato.
        """

        features_immagine_originale = [] #len 100
        features_immagine_scalata = [] #len 100
        features_immagine_scalata_25 = []
        features_matched_25 = []
        features_immagine_scalata_50 = []
        features_matched_50 = []
        features_immagine_scalata_75 = []
        features_matched_75 = []
        features_immagine_scalata_125 = []
        features_matched_125 = []
        features_immagine_scalata_150 = []
        features_matched_150 = []
        features_immagine_scalata_175 = []
        features_matched_175 = []
        features_immagine_scalata_200 = []
        features_matched_200 = []
        features_matched = [] #len 100
        tempo_features_detection_originale = [] #len 100
        tempo_features_detection_scalata = [] #len 100
        tempo_matching = [] #len 100

        media_fio = 0
        media_is = 0
        media_m = 0
        media_tfdo = 0
        media_tfds = 0
        media_tm = 0
        media_is25 = 0
        media_fm25 = 0
        media_is50 = 0
        media_fm50 = 0
        media_is75 = 0
        media_fm75 = 0
        media_is125 = 0
        media_fm125 = 0
        media_is150 = 0
        media_fm150 = 0
        media_is175 = 0
        media_fm175 = 0
        media_is200 = 0
        media_fm200 = 0

        data = {}
        data['risultato'] = []

        for j in range(len(self.img_list)):

            nome_immagine = self.img_list[j]

            temp = self.features_detection(nome_immagine, scale, algoritmo)

            features_immagine_originale.append(temp[0])
            features_immagine_scalata.append(temp[1])
            features_matched.append(temp[2])
            tempo_features_detection_originale.append(temp[3])
            tempo_features_detection_scalata.append(temp[4])
            tempo_matching.append(temp[5])
            features_immagine_scalata_25.append(temp[6])
            features_matched_25.append(temp[7])
            features_immagine_scalata_50.append(temp[8])
            features_matched_50.append(temp[9])
            features_immagine_scalata_75.append(temp[10])
            features_matched_75.append(temp[11])
            features_immagine_scalata_125.append(temp[12])
            features_matched_125.append(temp[13])
            features_immagine_scalata_150.append(temp[14])
            features_matched_150.append(temp[15])
            features_immagine_scalata_175.append(temp[16])
            features_matched_175.append(temp[17])
            features_immagine_scalata_200.append(temp[18])
            features_matched_200.append(temp[19])

            print('Features immagine originale: ' + str(features_immagine_originale[j]) + ', Features immagine scalata: ' + str(features_immagine_scalata[j]) + ', Match: ' + str(features_matched[j]))
            print('-------------------------------')

        for l in range(0, numero_immagini):
            media_fio = media_fio + features_immagine_originale[l]
            media_is = media_is + features_immagine_scalata[l]
            media_m = media_m + features_matched[l]
            media_tfdo = media_tfdo + tempo_features_detection_originale[l]
            media_tfds = media_tfds + tempo_features_detection_scalata[l]
            media_tm = media_tm + tempo_matching[l]
            media_is25 = media_is25 + features_immagine_scalata_25[l]
            media_fm25 = media_fm25 + features_matched_25[l]
            media_is50 = media_is50 + features_immagine_scalata_50[l]
            media_fm50 = media_fm50 + features_matched_50[l]
            media_is75 = media_is75 + features_immagine_scalata_75[l]
            media_fm75 = media_fm75 + features_matched_75[l]
            media_is125 = media_is125 + features_immagine_scalata_125[l]
            media_fm125 = media_fm125 + features_matched_125[l]
            media_is150 = media_is150 + features_immagine_scalata_150[l]
            media_fm150 = media_fm150 + features_matched_150[l]
            media_is175 = media_is175 + features_immagine_scalata_175[l]
            media_fm175 = media_fm175 + features_matched_175[l]
            media_is200 = media_is200 + features_immagine_scalata_200[l]
            media_fm200 = media_fm200 + features_matched_200[l]

        data['risultato'].append({
            'features_originale': media_fio / self.numero_immagini,
            'media_features': media_is / self.numero_immagini,
            'media_match': media_m / self.numero_immagini,
            'tempo_det_originale': media_tfdo / self.numero_immagini,
            'tempo_det_resized': media_tfds / self.numero_immagini,
            'media_tempo_matching': media_tm / self.numero_immagini,
            'media_is25': media_is25 / self.numero_immagini,
            'media_fm25': media_fm25 / self.numero_immagini,
            'media_is50': media_is50 / self.numero_immagini,
            'media_fm50': media_fm50 / self.numero_immagini,
            'media_is75': media_is75 / self.numero_immagini,
            'media_fm75': media_fm75 / self.numero_immagini,
            'media_is125': media_is125 / self.numero_immagini,
            'media_fm125': media_fm125 / self.numero_immagini,
            'media_is150': media_is150 / self.numero_immagini,
            'media_fm150': media_fm150 / self.numero_immagini,
            'media_is175': media_is175 / self.numero_immagini,
            'media_fm175': media_fm175 / self.numero_immagini,
            'media_is200': media_is200 / self.numero_immagini,
            'media_fm200': media_fm200 / self.numero_immagini
        })
        
        with open(self.result_path + algoritmo + '\\risultato.txt', 'w') as outfile:
            json.dump(data, outfile)

if __name__ == '__main__':
    """
        Test per calcolare la robustezza degli algoritmi di Features Detection
        AKAZE, BRISK, ORB e SIFT all'invarianza di rotazione e di scala.
        Nella cartella 'Risultati1/nome_algoritmo/risultato.txt' possono essere
        reperiti i risultati ottenuti per quel determinato algoritmo.
        
        img_path: percorso nel quale sono posizionate le immagini da analizzare
        img_list: lista contenente i nomi in ordine delle immagini, il ciclo for riempie la lista
        in questo modo ("0.tif", "1.tif", ..., "i.tif")
        scale: valori di scala (25, 50, 75, 125, 150, 175, 200)
        algoritmo: algoritmo selezionato per l'analisi (AKAZE, BRISK, ORB, SIFT)
    """
    result_path = 'C:\\Users\\Carlo\\Workplace\\Python\\Test\\Risultati1\\'
    img_path = 'C:\\Users\\Carlo\\Workplace\\Python\\Test\Immagini\\'
    img_list = []
    scale = [25, 50, 75, 125, 150, 175, 200]
    algoritmo = 'BRISK'
    numero_immagini = 5

    test = Test(result_path, img_path, img_list, numero_immagini)

    #Creo cartelle immagini scalate
    for i in range(0, len(scale)):
        os.makedirs(img_path + '\\' + str(scale[i]) + '\\', exist_ok=True) 

    #inserisco nella lista i nomi delle immagini ed eseguo le opportune rotazioni e ridimensionamenti 
    #la singola immagine
    for i in range(0, numero_immagini):
        nome_immagine = str(i) + '.tif'
        img_list.append(nome_immagine)
        test.resize_rotate_image(nome_immagine, scale)
    
    test.run(scale, algoritmo)

