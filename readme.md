Réseaux de neurones : concevoir, évaluer, optimiser, alléger et préparer du texte, et déployer son modèle

1. tp1_regression.ipynb — Prédire un nombre (La Régression)
   Objectif : Apprendre à prédire une valeur continue (courbe sinus, indice de progression du diabète sur le jeu de données médicales).

Notions clés : Sortie sans fonction d'activation, fonction de perte MSE (Mean Squared Error), métriques d'évaluation (MAE, RMSE, R²), et mise en évidence du sous-apprentissage et du surapprentissage.

2. tp2_classification_binaire.ipynb — Répondre par Oui ou Non (Classification Binaire)
   Objectif : Détecter si une tumeur du sein est maligne ou bénigne à partir de 569 échantillons cliniques.

Notions clés : Couche de sortie avec activation sigmoid, perte binary_crossentropy, construction d'une matrice de confusion (VP, VN, FP, FN), arbitrage entre Précision et Rappel, déplacement du seuil de décision et tracé de la courbe ROC / AUC.

3. tp3_multiclasses.ipynb — Reconnaître des Chiffres (Classification Multiclasses)
   Objectif : Classifier des images de chiffres manuscrits (0 à 9) issues de load_digits.

Notions clés : Sortie à 10 neurones avec activation softmax, perte sparse_categorical_crossentropy, lecture d'une matrice de confusion à 10 classes, et autopsie des erreurs commises par le modèle.

4. tp4_cnn.ipynb — Réseaux Convolutifs (CNN) pour la Vision par Ordinateur
   Objectif : Traiter des images en préservant leur structure spatiale 2D au lieu de les aplatir.

Notions clés : Couches de convolution (Conv2D), réduction spatiale (MaxPooling2D), calcul manuel du nombre de paramètres d'un filtre, et test de robustesse face au décalage d'un pixel (np.roll).

5. tp5_regularisation.ipynb & defi_moons_najia.ipynb — Surapprentissage, Bruit & Régularisation
   Objectif : Apprendre à maîtriser l'overfitting sur de petits jeux de données (ex: Défi des 600 points en croissants de lune make_moons).

Notions clés : Les 5 remèdes (Arrêt précoce EarlyStopping, Dropout, régularisation L2, réduction du réseau, augmentation de données), calcul de la marge de bruit statistique, et répétition des entraînements sur plusieurs graines (seed).

6. tp6_hyperparametres_compression.ipynb — Hyperparamètres & Compression (TFLite)
   Objectif : Optimiser les réglages d'un modèle et le rendre éligible à un déploiement embarqué (mobile/IoT).

Notions clés : Utilisation d'un jeu de validation indépendant (X_val), impact du taux d'apprentissage (lr), recherche aléatoire d'hyperparamètres, élagage (pruning) et quantification 8 bits avec TensorFlow Lite (passage de 89 Ko à 10 Ko sans perte de précision).

7. tp7_texte_nlp_2.ipynb — Traitement du Langage Naturel (NLP)
   Objectif : Transformer des données textuelles non structurées (2 400 résumés de films d'horreur ou comédies) en features numériques pour un classifieur.

Notions clés : Tokenisation (RegexpTokenizer), normalisation, suppression des mots vides (stop_words), comparaison entre le stemming NLTK (racines) et la lemmatisation spaCy (formes canoniques grammaticales), extraction d'entités nommées (NER), gestion des ambiguïtés, et vectorisation TF-IDF.

Notebook Défi Moons
Objectif : Résoudre un problème de classification binaire sur un jeu de données de 600 points non séparables linéairement (générés avec make_moons, bruit de 0.35). Il s'agissait de déterminer si un réseau de neurones pouvait surpasser une méthode simple (régression logistique) et de garantir rigoureusement ce résultat sur le plan statistique.


Boîte à outils & technologies utilisées
Manipulation & Calcul : NumPy, Pandas

Machine Learning Classique : Scikit-Learn (modèles linéaires, métriques, normalisation StandardScaler, vectorisation TfidfVectorizer)

Deep Learning : TensorFlow & Keras (architectures denses et convolutives, callbacks, quantification TFLite)

NLP : NLTK, spaCy (en_core_web_sm)
