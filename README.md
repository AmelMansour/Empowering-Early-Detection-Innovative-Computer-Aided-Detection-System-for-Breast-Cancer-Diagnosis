# Empowering-Early-Detection-Innovative-Computer-Aided-Detection-System-for-Breast-Cancer-Diagnosis

Le cancer du sein est l'une des principales causes de mortalité chez les femmes, avec 2,26 millions de nouveaux cas en 2022 (OMS). Bien que les méthodes de dépistage comme la mammographie aient progressé, elles souffrent de faux positifs/négatifs et d'une interprétation subjective. Ce mémoire propose un système de détection assistée par ordinateur (CAD) basé sur l'IA pour améliorer la précision du diagnostic via l'analyse automatisée d'images mammographiques.

# Objectifs

* Développer un modèle CNN optimisé pour classer les tumeurs (bénignes/malignes).

* Utiliser des algorithmes métaheuristiques (GA, PSO, SA, TS) et le framework Optuna pour optimiser les hyperparamètres.

* Évaluer les performances sur deux bases de données publiques : DDSM et INbreast.

* Comparer les résultats avec l'état de l'art pour valider l'efficacité clinique.

# Méthodologie
1. Données
DDSM : 13,128 images (PNG, 50μm/pixel).

INbreast : 7,632 images (DICOM, 70μm/pixel).

Prétraitement : Redimensionnement (256x256), CLAHE, Dénormalisation, Augmentation de données.

2. Modèles et Optimisation
CNN Personnalisé : Architecture avec couches convolutives, pooling, et dropout.

ResNet152 : Transfer Learning avec fine-tuning.

Optimisation :

Algorithmes métaheuristiques (GA, PSO, SA, TS).

Framework Optuna pour l'optimisation automatique.

3. Métriques d'Évaluation
Accuracy, Précision, Rappel, F1-Score, AUC-ROC.

Validation croisée stratifiée (5 folds).

📊 Résultats Clés
Modèle (Optimisé par Optuna)	Dataset	Accuracy	Précision	Rappel	F1-Score
CNN	DDSM	99.84%	99.86%	99.85%	99.85%
CNN	INbreast	97.77%	98.02%	98.73%	98.37%
ResNet152 (Fine-Tuning)	INbreast	72.14%	72.27%	94.78%	82.02%
Comparaison avec l'État de l'Art
Supériorité du CNN optimisé :

+1.03% vs. Mokni & Haoues (ResNet152 fine-tuned, 98.84%).

+26.78% vs. Ragab et al. (DCNN-SVM, 80.5%).

💡 Contributions Majeures
Optimisation Innovante : Combinaison d'algorithmes métaheuristiques et d'Optuna pour atteindre une précision record (99.84%).

Analyse Comparative : Validation rigoureuse sur deux datasets, mettant en évidence les forces/faiblesses des architectures (CNN vs. ResNet152).

Potentiel Clinique : Réduction des faux positifs/négatifs, facilitant un diagnostic plus fiable.
