# ChessBot

ChessBot est un projet Python qui automatise des mouvements sur le site chess.com en utilisant la vision par ordinateur et Stockfish pour le calcul des meilleurs coups.

## Fonctionnalités
- Recadrage et capture de l'échiquier.
- Détection des pièces avec OpenCV (traitement d'image).
- Calcul et suggestion de coups optimaux avec le moteur Stockfish.
- Automatisation des déplacements grâce à pyautogui.
- Mise à jour et suivi en temps réel de l'état de l'échiquier.

## Installation
1. Cloner le dépôt :
   ```bash
   git clone https://github.com/Zeffut/ChessBot
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

## Utilisation
- Exécuter le script principal :
   ```bash
   python main.py
   ```
- Vérifier la présence des fichiers images des pièces et de `win.png` dans le dossier adéquat.

## Configuration
- Modifier les coordonnées de recadrage de l'échiquier dans le fichier `main.py` si nécessaire.
- Ajuster les seuils et paramètres de détection pour optimiser le fonctionnement.

## Améliorations
- Ajout d'une interface graphique pour visualiser les coups.
- Ajout de l'enregistrement des données de partie.
- Ajout de la détection de l'ELO automatique.
- Ajout de la détection de la couleur automatique.

## Contribuer
Les contributions sont les bienvenues !  
Merci d'ouvrir une issue ou une pull request pour proposer des améliorations.
