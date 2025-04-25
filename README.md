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

## Améliorations
- Interface graphique pour visualiser les coups.
- Enregistrement des données de partie.
- Détection de l'ELO automatique.
- Détection de la couleur automatique.
- Détection automatique des coordonnées l'échiquier

## Avertissement
Ce projet n'est pas destiné à un usage commercial ou à une quelconque monétisation. Il a été créé à des fins éducatives et pour le plaisir de la programmation. L'utilisation de ce projet doit respecter les conditions d'utilisation des sites tiers, tels que chess.com.
