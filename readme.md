# Projet AI Frameworks

Dans ce projet, vous allez travailler sur des données<sup>[1](#myfootnote1)</sup>issues du site [Food.com](https://www.food.com/), un célèbre site de recettes de cuisines.   
![](img/food.png)
Les données, disponibles [ici](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_recipes.csv), contiennent des informations sur des recettes de cuisines ainsi que des interactions de plusieurs utilisateurs avec les recettes.   

## Consignes:
### Partie 1: Recommendations simples:
Dans __un notebook__
*   Présentez plusieurs stratégies de recommendations de recettes:
    *   Par popularité
    *   Selon les étapes de la recette (colonne steps)
    *   Selon la description de la recette 

    Pour chacune de ces métodes montrez quelques exemples des recommendations obtenues.

### Partie 2: Analyse de sentiments:
Dans __un notebook__
*   A partir de la note donnée par les utilisateurs definissez une nouvelle valiable sur le sentiment positif ou négatif d'un utilisateur vis-à-vis d'une recette.
Par exemple toutes les notes inférieures à 3 sont négatives et celles supérieurs sont positives.
* En vous inspirant du TP [_Text Cleaning and Vectorization_](https://colab.research.google.com/github/wikistat/AI-Frameworks/blob/master/Text/1_cleaning_vectorization.ipynb) et du TP [_Interpretability in Machine Learning_](https://colab.research.google.com/github/wikistat/AI-Frameworks/blob/website/code/interpretability/TP_interpretability_solution.ipynb), entrainez un modèle à prédire si un utilisateur a aimé ou non une recette à partir de son commentaire et utilisez la méthode lime pour visualiser les mots permettant de justifier la décision de votre modèle.

### Partie 3: Neural Collaborative Filtering:
*   Dans fichier ```model.py```, redefinissez la classe ```NCF``` (Neural Collaborative Filtering )présente dans le [TP sur les systèmes de recommendations](https://colab.research.google.com/github/wikistat/AI-Frameworks/blob/website/code/recommender_systems/INSA_Reco_solution.ipynb).
* Dans un fichier ```main.py```, implémentez un code permettant d'entrainer un modèle de Neural Collaborative Filtering à prédire les notes d'un utilisateur.
Ce fichier sera exécuté comme un script et devra:
    * Récupérer le chemin du dataset de train et du dataset de test en arguments de la commande qui exécutera le script 
    *   Utiliser la classe ```NCF``` présente dans le fichier ```model.py``` pour créer un modèle.
    *   Mettre le dataset de train au format Pytorch (inspirez vous de la classe ```Ratings_Dataset``` du TP sur les systèmes de recommendations)
    *   Entrainer le modèle sur le dataset de train
    *   Afficher les predictions pour les 10 interactions du dataset de test.

### Partie 4 Github:
*   Faites un fork de ce repo Git.
*   Modifiez le pour qu'il contiennes les notebooks des parties 1 et 2
*   Rajouter les fichiers ```model.py``` et ```main.py```
*   Modifiez le readme pour qu'il affiche votre nom et la commande permetttant de lancer correctement le script ```main.py``` (ne mettez pas les fichiers de données dans le repo github!)


 

<a name="myfootnote1">1</a>: Les données ont été recoltées pour l'article suivant:  
 [Generating Personalized Recipes from Historical User Preferences
Bodhisattwa Prasad Majumder*, Shuyang Li*, Jianmo Ni, Julian McAuley
EMNLP, 2019](https://www.aclweb.org/anthology/D19-1613/)