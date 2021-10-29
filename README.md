### Descritption

Ce programme permet de réaliser l'apprentissage du modèle UNET-3D pour la labellisation automatique des sillons corticaux (modèle dans brainvisa).
Deux options d'apprentissage sont possibles : un apprentissage complet ou un transfert d'apprentissage à partir d'un modèle déjà entraîné.
Dans le second cas, seul la dernière couche convolutionnelle est entraînée (final_conv). \

Le dossier comporte plusieurs fichiers:
* Le fichier `pattern_class.py` qui contient la classe `UnetPatternSulciLabelling`
* Les fichiers `transfer_learning.py` et `training.py` qui comportent les deux classes avec toutes les fonctions qui permettent de réaliser l'apprentissage. Ces deux classes héritent de la classe `UnetPatternSulciLabelling`. Ces classes sont des variantes de la classe `UnetSulciLabelling` de brainvisa.
* Le fichier `main.py` qui permet de regrouper les graphes de la cohorte choisie, puis de lancer l'entraînement
* Le fichier `dataset.py` est une variante du fichier présent dans brainvisa avec la possibilité de fixer la taille d'image
* Les fichiers `divide_lr.py` et `fine_tunning.py` qui sont des outils pour les classes d'apprentissage


### Fonctionnement
* 1ère étape : chargement des données et du modèle
* 2ème étape : apprentissage du modèle
* 3ème étape : applications du cutting threshold (si l'option est choisie)
* 4ème étape : enregistrement des résultats

###### Entrées
Il faut créer un fichier _parameters.json_ de la forme : \

{\
"**working_path**": $PATH$, *str: chemin enregistrement des résultats*\
"**learning**": "transfer_learning" , *str: type d'apprentissage*\
"**cohort_name**": "Chimp_3T", *str: le nom de la cohort*\
"**hemi**": "L", *str: L ou R pour l'hémisphère*\
"**path_to_cohort**": $PATH$, *str: chemin où est stocké le fichier json de la cohort*\
"**translation_file**": $PATH$, *fichier .trl*\
"**batch_size**": 1, *int*\
"**lr**": 1e-2, *float: learning rate*\
"**momentum**": 0.9, *float*\
"**n_cvinner**": 3, *int: nombre de folder pour la cross validation*\
"**n_epochs**": 5, *int: nombre d'epochs*\
"**th_range**": [50, 100, 150], *list of int, liste des cutting threshold à appliquer* \
"**dict_model**": { "name": "unknown_name", *str: nom du modèle* \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; "training_layers": ["final_conv"], *list of str : couches à entraîner pendant l'apprentissage* \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; "fine_tunning_layers": ['decoders.2', 'decoders.1', 'decoders.0'],  *list of str : couches à entraîner pendant le fine tunning* \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; }  \
"**dict_trained_model**": {"in_channels": 1, *int*\
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp; "out_channels": 56, *int ou str, dans ce cas, il s'agit du path vers un fichier params.json*\
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp; "conv_layer_order": 'crg', \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp; "interpolate": True, \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp; "final_sigmoid": False, \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp; "init_channel_number" : 64, \
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;} \
} \
\
Puis, il faut lancer la commande `python main.py` ! \
L'option `-p` permet de spécifier le chemin vers le _fichier parameters.json_ s'il ne se trouve pas dans le même dossier que le programme `main.py`.  \
\
NB : Les dictionnaire dict_model et dict_trained_model n'ont pas besoin de présenter toutes les clés, dans ce cas, les valeurs par défaut sont implémentées (indiquérs juste au-dessous).
Dans le cas de l'apprentissage complet, il n'y a pas besoin de spécifier dict_trained_model et les clés "training_layers" et "fine_tunning_layers" du dict_model. \ 

###### Sorties
Le programme enregistre dans les répertoires :
* _working_path/data_ : un fichier _cohort_name.json_ qui enregistre les variables dict_sulci, dict_names et sulci_side_list
* _working_path/results_ : un fichier _model_name.json_ qui enregistre les paramètres d'entraînement (batch size, momentum, lr, les graphs du train et val test), la loss et l'accuracy de chaque epoch sur le train et val test, la meilleure epoch, la meilleure accuracy et le scores des cutting thresholds
* _working_path/models_ : un fichier _mode_name_cvx_model.mdsm_ (x le numéro de l'entraînement de la cross validation) qui sauvegarde les poids du modèle (state_dict) et un fichier _model_name_cvx_params.json_ qui enregistre les variables dict_names, dict_sulci, sulci_side_list et le meilleur cutting_threshold 
* _working_path/tensorboard_ : un dossier qui permet d'afficher sur tensorboard la loss et l'accuracy sur les train et val sets pour chaque entraînement. \
\
NB: pour utiliser tensorboard, il faut :
1. installer tensorboard dans l'image singularity de brainvisa : `pip3 install tensorboard`
2. lancer le main dans le package tensorboard (pour voir le chemin, lancer la commande `pip3 show tensorboard`) avec la commande `python3 main.py --logdir $PATH$` avec path le chemin vers le dossier du modèle dans le répertoire tensorboard.
