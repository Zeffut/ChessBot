# ChessBot ♟️

ChessBot est un projet Python qui automatise des mouvements sur le site chess.com en utilisant la vision par ordinateur et Stockfish pour le calcul des meilleurs coups. Ce projet a été conçu pour explorer les capacités de l'automatisation et de l'intelligence artificielle dans le domaine des échecs.

## Fonctionnalités ✨
- **Recadrage et capture de l'échiquier** : 📸 Capture d'écran de l'échiquier pour analyse.
- **Détection des pièces** : 🔍 Utilisation d'OpenCV pour identifier les pièces sur l'échiquier.
- **Calcul des coups optimaux** : ♟️ Intégration du moteur Stockfish pour suggérer les meilleurs coups.
- **Automatisation des déplacements** : 🤖 Utilisation de pyautogui pour exécuter les mouvements sur chess.com.
- **Mise à jour en temps réel** : ⏱️ Suivi dynamique de l'état de l'échiquier.

## Prérequis 🛠️
Avant de commencer, assurez-vous d'avoir les éléments suivants :
- 🐍 Python 3.8 ou supérieur installé sur votre machine.
- 📦 pip pour gérer les dépendances Python.
- ⚙️ Stockfish (inclus dans le projet).
- 🖼️ OpenCV et pyautogui (installés via `requirements.txt`).

## Installation 🚀
1. **Cloner le dépôt** :
   ```bash
   git clone https://github.com/Zeffut/ChessBot
   ```
2. **Naviguer dans le répertoire du projet** :
   ```bash
   cd ChessBot
   ```
3. **Installer les dépendances** :
   ```bash
   pip install -r requirements.txt
   ```

## Structure du projet 🗂️
Voici une vue d'ensemble des principaux fichiers et dossiers :
- `chess_bot.py` : Script principal pour exécuter le bot.
- `camera.py` : Script permettant d'analyser le gameplay d'une camera
- `overlay.py` : Script permettant de gérer un overlay
- `cheat.py` : Script permettant d'afficher les coups sans les jouers pour permettre au joueur de triché correctement. (ATTENTION SCRIPT EN BETA)
- `requirements.txt` : Liste des dépendances Python nécessaires.
- `ressources/` : Contient les images des pièces et autres ressources graphiques.
- `stockfish/` : Contient le moteur Stockfish et ses fichiers associés.
- `testing/` : Scripts pour tester et valider les fonctionnalités du projet.

## Utilisation 🕹️
1. **Exécuter le script principal** :
   ```bash
   python chess_bot.py
   ```
2. **Configurer les paramètres** : Assurez-vous que l'échiquier est visible à l'écran et que les coordonnées sont correctement détectées.
3. **Suivre les instructions** : Le bot détectera automatiquement les pièces et suggérera les meilleurs coups.

## Améliorations futures 🔮
- 🖥️ Interface graphique pour visualiser les coups.
- 📝 Enregistrement des données de partie.
- 🤔 Détection automatique de l'ELO

## Avertissement ⚠️
Ce projet n'est pas destiné à un usage commercial ou à une quelconque monétisation. Il a été créé à des fins éducatives et pour le plaisir de la programmation. L'utilisation de ce projet doit respecter les conditions d'utilisation des sites tiers, tels que chess.com.
