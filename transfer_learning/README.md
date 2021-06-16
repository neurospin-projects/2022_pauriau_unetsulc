### Descritption

Ce programme permet de réentrainer uniquement la dernière couche de neurones (Conv3D) sur une nouvelle base des données.
Les poids et biais des autres couches restent identiques au modèle entrainé sur les cerveaux humains.

### Fonctionnement

* 1ère étape : chargement des données et du modèles
* 2ème étape : entrainement en cross validation afin de déterminer les performances du modèle

###### Entrées
Il faut placer un fichier parameters.json dans le même dossier que le main, de la forme : \
{"working_path": $PATH$, *str: chemin enregistrement des résultats*\
"cohort_name": "Chimp_3T_short", *str: le nom de la cohort*\
"hemi": "L",*str: L ou R pour l'hémisphère*\
"path_to_cohort": $PATH$, *str: chemin où est stocké le fichier json de la cohort*\
"translation_file": $PATH$, *fichier .trl*\
"batch_size": 1,*int*\
"lr": 1e-2,*int: learning rate*\
"momentum": 0.9, *float*\
"n_cvinner": 3, *int: nombre de folder pour la cross validation*\
"n_epochs": 5} *int: nombre d'epochs*\
Puis, lancer le programme main.py !

###### Sorties
Le programme enregistre :
* un fichier data.json avec dict_sulci, dict_names et sulci_side_list
* un fichier results.json avec la loss, l'accuracy du modèle
* un dossier model, enregistrement des poids du modèles (state_dict)

