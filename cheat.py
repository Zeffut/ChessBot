import sys
import os
import time
import random
import cv2
import numpy as np
import chess
import chess.engine
import tkinter as tk
import pyautogui
from PIL import Image, ImageDraw, ImageFont

debug = False
debug_counter = 0

# --- Imports spécifiques Mac ---
if sys.platform == "darwin":
    import subprocess
    from Quartz import CGDisplayBounds, CGMainDisplayID, CGWindowListCreateImage, kCGWindowListOptionOnScreenOnly, kCGNullWindowID, kCGWindowImageDefault
    import threading
    import objc
    from PIL import ImageDraw

# --- Paramètres dépendants de l'OS ---
if sys.platform == "win32":
    STOCKFISH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ressources/stockfish/stockfish-windows-x86-64-avx2.exe")
    def take_screenshot(output_path="screenshot.png"):
        screenshot = pyautogui.screenshot()
        screenshot.save(output_path)
elif sys.platform == "darwin":
    STOCKFISH_PATH = "stockfish"
    def take_screenshot(output_path="screenshot.png"):
        subprocess.run(["screencapture", "-x", output_path])
else:
    raise NotImplementedError("OS non supporté")

# --- Fonctions communes (fusionnées) ---

def detect_chessboard(image_path):
    """
    Détecte l'emplacement de l'échiquier dans une image.
    
    Args:
        image_path (str): Chemin vers l'image à analyser
        
    Returns:
        tuple: ((x1, y1), (x2, y2)) où:
               (x1, y1) sont les coordonnées du coin supérieur gauche
               (x2, y2) sont les coordonnées du coin inférieur droit
               None si aucun échiquier n'est détecté
    """
    # Lecture de l'image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Impossible de charger l'image")
    
    # Conversion en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Application d'un flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Détection des contours avec la méthode de Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilatation pour renforcer les contours
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Recherche des contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Tri des contours par aire
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    for contour in contours:
        # Approximation du contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Si le contour a 4 sommets (rectangle), c'est potentiellement l'échiquier
        if len(approx) == 4:
            # Vérification des proportions (un échiquier est carré)
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            
            # Si le ratio largeur/hauteur est proche de 1 (tolérance de 20%)
            if 0.8 <= aspect_ratio <= 1.2:
                # Vérification de la taille minimale
                if w > 100 and h > 100:
                    # Retourne les coordonnées des coins supérieur gauche et inférieur droit
                    return ((x, y), (x + w, y + h))
    return None

def crop_board(image_path, top_left, bottom_right, output_path="cropped_board.png"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image : {image_path}")
        return
    x1, y1 = top_left
    x2, y2 = bottom_right
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)

def preprocess_image(image):
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    return gray

def detect_piece(square_image, piece_image, ratio_threshold=0.75, match_threshold=10):
    gray_square = preprocess_image(square_image)
    gray_piece = preprocess_image(piece_image)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray_piece, None)
    kp2, des2 = orb.detectAndCompute(gray_square, None)
    if des1 is None or des2 is None:
        return False
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in raw_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
    return len(good_matches) > match_threshold

def remove_unwanted_colors(image, tolerance=20):
    unwanted_colors = [
        np.array([90, 148, 122]),
        np.array([211, 236, 235]),
        np.array([140, 244, 248]),
        np.array([92, 204, 192]),
        np.array([68, 204, 192])
    ]
    for col in unwanted_colors:
        lower = np.clip(col - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(col + tolerance, 0, 255).astype(np.uint8)
        mask = cv2.inRange(image, lower, upper)
        image[mask != 0] = [0, 0, 0]
    return image

def analyze_board_orb(cropped_board_path, pieces_folder):
    global debug_counter
    board_image = cv2.imread(cropped_board_path)
    if board_image is None:
        print(f"Impossible de charger l'image : {cropped_board_path}")
        return
    board_image = remove_unwanted_colors(board_image)
    height, width = board_image.shape[:2]
    square_height = height // 8
    square_width = width // 8
    pieces = {}
    for filename in os.listdir(pieces_folder):
        if filename.endswith(".png"):
            piece_name = os.path.splitext(filename)[0]
            piece_image = cv2.imread(os.path.join(pieces_folder, filename), cv2.IMREAD_UNCHANGED)
            if piece_image is None:
                continue
            if len(piece_image.shape) == 3 and piece_image.shape[2] == 4:
                piece_image = cv2.cvtColor(piece_image, cv2.COLOR_BGRA2BGR)
            piece_resized = cv2.resize(piece_image, (square_width, square_height), interpolation=cv2.INTER_AREA)
            piece_gray = preprocess_image(piece_resized)
            pieces[piece_name] = piece_gray
    probability_threshold = 0.5
    board_state = []
    for row in range(8):
        board_row = []
        for col in range(8):
            y1, y2 = row * square_height, (row + 1) * square_height
            x1, x2 = col * square_width, (col + 1) * square_width
            square = board_image[y1:y2, x1:x2]
            square_gray = preprocess_image(square)
            best_match = "empty"
            best_prob = 0.0
            for piece_name, piece_template in pieces.items():
                if square_gray.shape != piece_template.shape:
                    piece_template = cv2.resize(piece_template, (square_gray.shape[1], square_gray.shape[0]))
                res = cv2.matchTemplate(square_gray, piece_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_prob:
                    best_prob = max_val
                    best_match = piece_name
            if best_prob < probability_threshold:
                best_match = "empty"
            else:
                piece_color = detect_piece_color(square)
                if piece_color:
                    best_match += piece_color
            board_row.append(best_match)
        board_state.append(board_row)
    # --- DEBUG OVERLAY ---
    if debug:
        from PIL import ImageDraw, ImageFont
        pil_img = Image.fromarray(cv2.cvtColor(board_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        try:
            font = ImageFont.truetype("arial.ttf", int(square_height/4))
        except:
            font = ImageFont.load_default()
        for row in range(8):
            for col in range(8):
                name = board_state[row][col]
                x = col * square_width + 5
                y = row * square_height + 5
                draw.text((x, y), name, fill=(255,0,0), font=font)
        debug_counter += 1
        pil_img.save(f"debug_board_{debug_counter}.png")
    # --- END DEBUG ---
    return board_state

def detect_piece_color(square_image):
    height, width = square_image.shape[:2]
    center_y, center_x = int(height - height*0.2), width // 2
    offset = 2
    region = square_image[center_y - offset:center_y + offset + 1, center_x - offset:center_x + offset + 1]
    region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(region_hsv.reshape(-1, 3), axis=0)
    h, s, v = avg_hsv
    if v > 150 and s < 50:
        return "B"
    elif v < 100:
        return "N"
    else:
        return None

def board_state_to_fen(board_state, active_color):
    piece_to_fen = {
        "roi": "k", "reine": "q", "tour": "r", "fou": "b", "cavalier": "n", "pion": "p"
    }
    fen_rows = []
    
    # Si on joue les noirs, on inverse l'échiquier
    if active_color == "b":
        board_state = [row[::-1] for row in board_state[::-1]]
    
    for row in board_state:
        empty_count = 0
        fen_row = ""
        for square in row:
            if square == "empty":
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                piece = square[:-1]
                color = square[-1]
                fen_piece = piece_to_fen.get(piece.lower())
                if fen_piece is not None:  # Only add piece if it's a known piece type
                    if color == "B":
                        fen_row += fen_piece.upper()
                    else:
                        fen_row += fen_piece.lower()
                else:
                    empty_count += 1  # Treat unknown pieces as empty squares
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    fen = "/".join(fen_rows) + f" {active_color} - - 0 1"
    return fen

def get_best_move_from_stockfish(fen, stockfish_path=STOCKFISH_PATH, max_retries=3):
    for i in range(max_retries):
        try:
            with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
                engine.configure({"Skill Level": 20})
                board = chess.Board(fen)
                result = engine.play(board, chess.engine.Limit(time=2.0 if sys.platform == "win32" else 0.5))
                return result.move
        except chess.engine.EngineTerminatedError:
            print(f"Attempt {i+1}/{max_retries} failed: engine terminated unexpectedly")
    return None

def explain_move(move, board_state):
    start_square = (8 - int(move[1]), ord(move[0]) - ord('a'))
    end_square = (8 - int(move[3]), ord(move[2]) - ord('a'))
    piece = board_state[start_square[0]][start_square[1]]
    piece_name = piece[:-1] if piece != "empty" else "aucune pièce"
    start_square_name = move[:2]
    end_square_name = move[2:]
    return f"Déplacez {piece_name} de {start_square_name} à {end_square_name}."

def perform_drag_and_drop_with_pyautogui(move, board_top_left, square_size, scale_factor, button="left", active_color="w"):
    start_col = ord(move[0]) - ord('a')
    start_row = 8 - int(move[1])
    end_col = ord(move[2]) - ord('a')
    end_row = 8 - int(move[3])
    
    # Si on joue les noirs, on inverse les coordonnées
    if active_color == "b":
        start_col = 7 - start_col
        start_row = 7 - start_row
        end_col = 7 - end_col
        end_row = 7 - end_row
    
    # Calcul des coordonnées des cases
    start_x = board_top_left[0] + start_col * square_size + square_size // 2
    start_y = board_top_left[1] + start_row * square_size + square_size // 2
    end_x = board_top_left[0] + end_col * square_size + square_size // 2
    end_y = board_top_left[1] + end_row * square_size + square_size // 2
    
    # Ajustement des coordonnées selon le facteur d'échelle
    start_x = int(start_x / scale_factor)
    start_y = int(start_y / scale_factor)
    end_x = int(end_x / scale_factor)
    end_y = int(end_y / scale_factor)
    
    pyautogui.moveTo(start_x, start_y, duration=0.1)
    pyautogui.mouseDown(button=button)
    pyautogui.moveTo(end_x, end_y, duration=0.1)
    pyautogui.mouseUp(button=button)

def calculate_scale_factor():
    screen_width, screen_height = pyautogui.size()
    screenshot = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape
    scale_factor_x = image_width / screen_width
    scale_factor_y = image_height / screen_height
    if abs(scale_factor_x - scale_factor_y) > 0.01:
        print("Attention : Les facteurs d'échelle X et Y diffèrent légèrement.")
    return scale_factor_x

def diff_board_states(prev_state, curr_state):
    differences = []
    for i in range(8):
        for j in range(8):
            if prev_state[i][j] != curr_state[i][j]:
                differences.append(((i, j), prev_state[i][j], curr_state[i][j]))
    return differences

def update_board_state(state, move_str):
    import copy
    new_state = copy.deepcopy(state)
    start_row = 8 - int(move_str[1])
    start_col = ord(move_str[0]) - ord('a')
    end_row = 8 - int(move_str[3])
    end_col = ord(move_str[2]) - ord('a')
    piece = new_state[start_row][start_col]
    new_state[start_row][start_col] = "empty"
    new_state[end_row][end_col] = piece
    return new_state

def detect_win(cropped_board_path, win_image_path="ressources/win.png", threshold=0.8):
    board_img = cv2.imread(cropped_board_path)
    win_img = cv2.imread(win_image_path)
    win_bot_img = cv2.imread("ressources/win_bot.png")
    end_img = cv2.imread("ressources/end.png")
    
    if board_img is None:
        return False, None
        
    # Détection de l'image end.png
    if end_img is not None:
        res = cv2.matchTemplate(board_img, end_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            # Si end.png est détecté, on vérifie si c'est une victoire ou une défaite
            if win_img is not None:
                res = cv2.matchTemplate(board_img, win_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val >= threshold:
                    return True, "victory"
            if win_bot_img is not None:
                res = cv2.matchTemplate(board_img, win_bot_img, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val >= threshold:
                    return True, "victory"
            return True, "defeat"
    
    # Détection des images de victoire classiques
    if win_img is not None:
        res = cv2.matchTemplate(board_img, win_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return True, "victory"
    if win_bot_img is not None:
        res = cv2.matchTemplate(board_img, win_bot_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return True, "victory"
            
    return False, None

def detect_active_color(board_state):
    """
    Détecte la couleur active en se basant sur la position du roi le plus bas sur l'échiquier.
    Le roi le plus bas est celui qui est le plus proche du bord inférieur.
    Si le roi blanc est plus bas, c'est aux blancs de jouer.
    Si le roi noir est plus bas, c'est aux noirs de jouer.
    Returns:
        str: 'w' si c'est aux blancs de jouer, 'b' si c'est aux noirs
    """
    white_king_pos = None
    black_king_pos = None
    
    # Parcourir l'échiquier pour trouver les positions des rois
    for row in range(8):
        for col in range(8):
            piece = board_state[row][col]
            if piece.startswith("roi"):
                if piece.endswith("B"):
                    white_king_pos = row
                elif piece.endswith("N"):
                    black_king_pos = row
    
    # Si un des rois n'est pas trouvé, retourner une valeur par défaut
    if white_king_pos is None or black_king_pos is None:
        return 'w'
    
    # Le roi le plus bas (row plus grand) indique que c'est à sa couleur de jouer
    return 'w' if white_king_pos > black_king_pos else 'b'

def display_move_on_screen(move, board_top_left, square_size, scale_factor, active_color="w", root=None, canvas=None):
    """
    Affiche le coup suggéré directement sur l'écran dans une fenêtre transparente et click-through.
    Si root et canvas sont fournis, met à jour l'affichage existant.
    Sinon, crée une nouvelle fenêtre.
    """
    import tkinter as tk
    from PIL import Image, ImageTk, ImageDraw

    start_col = ord(move[0]) - ord('a')
    start_row = 8 - int(move[1])
    end_col = ord(move[2]) - ord('a')
    end_row = 8 - int(move[3])
    
    # Si on joue les noirs, on inverse les coordonnées
    if active_color == "b":
        start_col = 7 - start_col
        start_row = 7 - start_row
        end_col = 7 - end_col
        end_row = 7 - end_row
    
    # Calcul des coordonnées des cases
    start_x = board_top_left[0] + start_col * square_size + square_size // 2
    start_y = board_top_left[1] + start_row * square_size + square_size // 2
    end_x = board_top_left[0] + end_col * square_size + square_size // 2
    end_y = board_top_left[1] + end_row * square_size + square_size // 2
    
    # Ajustement des coordonnées selon le facteur d'échelle
    start_x = int(start_x / scale_factor)
    start_y = int(start_y / scale_factor)
    end_x = int(end_x / scale_factor)
    end_y = int(end_y / scale_factor)
    
    # Si la fenêtre n'existe pas encore, la créer
    if root is None or canvas is None:
        # Création de la fenêtre principale
        root = tk.Tk()
        root.attributes('-topmost', True)  # Toujours au-dessus
        root.overrideredirect(True)  # Supprime la barre de titre

        # Définir la couleur de fond et la couleur transparente
        bg_color = 'black'  # Fond noir
        root.attributes('-transparentcolor', bg_color)  # Rendre le noir transparent

        # Configuration de la fenêtre
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.geometry(f"{screen_width}x{screen_height}+0+0")

        # Création du canvas
        canvas = tk.Canvas(root, width=screen_width, height=screen_height, highlightthickness=0, bg=bg_color)
        canvas.pack(fill=tk.BOTH, expand=True)

        # Ajout d'un bouton de fermeture pour le débogage
        close_button = tk.Button(root, text="Fermer", command=root.destroy)
        close_button.pack()

    # Effacer le contenu précédent
    canvas.delete("all")  # Efface tous les anciens coups
    
    # Couleurs pour les cercles et la flèche
    circle_color = "#00FF00"  # Vert
    arrow_color = "#FF0000"   # Rouge
    
    # Dessin des cercles
    canvas.create_oval(start_x-10, start_y-10, start_x+10, start_y+10, fill=circle_color, outline="black", width=2)
    canvas.create_oval(end_x-10, end_y-10, end_x+10, end_y+10, fill=circle_color, outline="black", width=2)
    
    # Dessin de la flèche
    canvas.create_line(start_x, start_y, end_x, end_y, fill=arrow_color, width=3, arrow=tk.LAST)
    
    # Mettre à jour l'interface
    root.update()

    return root, canvas  # Ajoutez cette ligne pour retourner root et canvas

if __name__ == "__main__":
    canvas = None
    root = tk.Tk()
    ascii_art = [
        "  /$$$$$$  /$$",
        " /$$__  $$| $$",
        "| $$  \\__/| $$$$$$$   /$$$$$$   /$$$$$$$ /$$$$$$$",
        "| $$      | $$__  $$ /$$__  $$ /$$_____//$$_____/",
        "| $$      | $$  \\ $$| $$$$$$$$|  $$$$$$|  $$$$$$ ",
        "| $$    $$| $$  | $$| $$_____/ \\____  $$\\____  $$",
        "|  $$$$$$/| $$  | $$|  $$$$$$$ /$$$$$$$//$$$$$$$/",
        " \\______/ |__/  |__/ \\_______/|_______/|_______/",
        "                                                    ",
        "                    by Zeffut                        "
    ]
    
    for line in ascii_art:
        print(line)

    time.sleep(5)
    
    scale_factor = calculate_scale_factor()

    screenshot_path = "screenshot.png"
    take_screenshot(screenshot_path)
    result = detect_chessboard(screenshot_path)
    if result is None:
        print("Error: Could not detect chessboard in screenshot. Please ensure the chessboard is visible and properly captured.")
        exit(1)
    top_left, bottom_right = result
    crop_board(screenshot_path, top_left, bottom_right, "cropped_board.png")    
    state = analyze_board_orb("cropped_board.png", "ressources/pieces")

    if state:
        active_color = detect_active_color(state)
        if active_color == "w":
            previous_state = [
                ['tourN', 'cavalierN', 'fouN', 'reineN', 'roiN', 'fouN', 'cavalierN', 'tourN'],
                ['pionN', 'pionN', 'pionN', 'pionN', 'pionN', 'pionN', 'pionN', 'pionN'],
                ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
                ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
                ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
                ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
                ['pionB', 'pionB', 'pionB', 'pionB', 'pionB', 'pionB', 'pionB', 'pionB'],
                ['tourB', 'cavalierB', 'fouB', 'reineB', 'roiB', 'fouB', 'cavalierB', 'tourB']
            ]
        else:
            previous_state = [
                ['tourB', 'cavalierB', 'fouB', 'reineB', 'roiB', 'fouB', 'cavalierB', 'tourB'],
                ['pionB', 'pionB', 'pionB', 'pionB', 'pionB', 'pionB', 'pionB', 'pionB'],
                ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
                ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
                ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
                ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
                ['pionN', 'pionN', 'pionN', 'pionN', 'pionN', 'pionN', 'pionN', 'pionN'],
                ['tourN', 'cavalierN', 'fouN', 'reineN', 'roiN', 'fouN', 'cavalierN', 'tourN']
            ]
        print(f"Couleur active détectée : {'Blanc' if active_color == 'w' else 'Noir'}")
        if active_color == "w":
            print("Joueur blanc - coup joué dès le départ")
            fen_A = board_state_to_fen(state, active_color)
            best_move_A = get_best_move_from_stockfish(fen_A)
            if best_move_A is not None:
                move_str = str(best_move_A)
                explanation = explain_move(move_str, state)
                print("Coup suggéré :", explanation)
                square_size = (bottom_right[0] - top_left[0]) // 8
                # Initialiser root et canvas avec un coup vide
                root, canvas = display_move_on_screen("a1a1", top_left, square_size, scale_factor, active_color=active_color)
                # Afficher le vrai coup
                display_move_on_screen(move_str, top_left, square_size, scale_factor, active_color=active_color, root=root, canvas=canvas)
            print("Début de la partie...")

    while True:
        screenshot_path = "screenshot.png"
        take_screenshot(screenshot_path)
        crop_board(screenshot_path, top_left, bottom_right, "cropped_board.png")
        game_over, result = detect_win("cropped_board.png")
        if game_over:
            if result == "victory":
                print("Partie Gagnée ! Arrêt du script.")
            else:
                print("Partie Perdue ! Arrêt du script.")
            os.remove("cropped_board.png")
            os.remove("screenshot.png")
            sys.exit()
        state = analyze_board_orb("cropped_board.png", "ressources/pieces")
        if state:
            if previous_state is None:
                previous_state = state
            elif state != previous_state:
                diffs = diff_board_states(previous_state, state)
                for pos, prev, curr in diffs:
                    row, col = pos
                previous_state = state
                fen_A = board_state_to_fen(state, active_color)
                best_move_A = get_best_move_from_stockfish(fen_A)
                if best_move_A is None:
                    print("Aucun coup suggéré par Stockfish.")
                    continue
                move_str = str(best_move_A)
                explanation = explain_move(move_str, state)
                print("Coup suggéré :", explanation)
                square_size = (bottom_right[0] - top_left[0]) // 8
                display_move_on_screen(move_str, top_left, square_size, calculate_scale_factor(), active_color=active_color, root=root, canvas=canvas)
                previous_state = update_board_state(previous_state, move_str)
