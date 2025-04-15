import cv2
import numpy as np
import os
import time
import subprocess
import chess
import chess.engine
import pyautogui
import random
import sys
from Quartz import CGDisplayBounds, CGMainDisplayID, CGWindowListCreateImage, kCGWindowListOptionOnScreenOnly, kCGNullWindowID, kCGWindowImageDefault
import threading
import objc
from PIL import Image, ImageDraw
from AppKit import (
    NSApplication, NSWindow, NSView,
    NSBorderlessWindowMask, NSBackingStoreBuffered,
    NSScreen, NSColor
)
from Foundation import NSRect, NSPoint, NSSize

# Variables globales pour l'overlay
overlay_window = None
line_view = None

class LineView(NSView):
    def initWithPoints_(self, points, is_point=False):
        screen_frame = NSScreen.mainScreen().frame()
        self = objc.super(LineView, self).initWithFrame_(screen_frame)
        self.p1, self.p2 = points
        self.is_point = is_point
        return self

    def drawRect_(self, rect):
        if self.is_point:
            circle_path = objc.lookUpClass('NSBezierPath').bezierPathWithOvalInRect_(
                NSRect((self.p1.x - 10, self.p1.y - 10), (20, 20))
            )
            NSColor.redColor().set()
            circle_path.fill()
        else:
            NSColor.redColor().set()
            path = objc.lookUpClass('NSBezierPath').alloc().init()
            path.setLineWidth_(3)
            path.moveToPoint_(self.p1)
            path.lineToPoint_(self.p2)
            path.stroke()
            
            for point in [self.p1, self.p2]:
                circle_path = objc.lookUpClass('NSBezierPath').bezierPathWithOvalInRect_(
                    NSRect((point.x - 4, point.y - 4), (8, 8))
                )
                NSColor.blueColor().set()
                circle_path.fill()

def init_overlay_window():
    global overlay_window, line_view
    print("Initializing overlay window...")
    app = NSApplication.sharedApplication()
    app.activateIgnoringOtherApps_(True)

    screen_frame = NSScreen.mainScreen().frame()
    print(f"Screen frame: {screen_frame}")
    window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        screen_frame,
        NSBorderlessWindowMask,
        NSBackingStoreBuffered,
        False
    )

    window.setLevel_(float(1000))
    window.setOpaque_(False)
    window.setBackgroundColor_(NSColor.clearColor())
    window.setIgnoresMouseEvents_(True)
    window.setAlphaValue_(1.0)
    window.setHasShadow_(False)
    window.setCollectionBehavior_(1 << 3)

    view = LineView.alloc().initWithPoints_([NSPoint(0, 0), NSPoint(0, 0)])
    window.setContentView_(view)
    window.makeKeyAndOrderFront_(None)
    overlay_window = window
    line_view = view
    
    print("Overlay window initialized successfully")
    return view

def create_point(x, y):
    global line_view
    print(f"Creating point at: ({x}, {y})")
    if not overlay_window:
        print("Overlay window not initialized, initializing now...")
        line_view = init_overlay_window()
    
    point = (x, y)
    print(f"Transformed point: {point}")
    
    new_view = LineView.alloc().initWithPoints_([NSPoint(point[0], point[1]), NSPoint(0, 0)], is_point=True)
    overlay_window.setContentView_(new_view)
    line_view = new_view
    
    line_view.setNeedsDisplay_(True)
    overlay_window.display()
    overlay_window.orderFront_(None)
    print("Point created successfully")

def update_line(p1, p2):
    global line_view
    print(f"Updating line with points: p1={p1}, p2={p2}")
    if not overlay_window:
        print("Overlay window not initialized, initializing now...")
        line_view = init_overlay_window()
    
    p1t = (p1[0], p1[1])
    p2t = (p2[0], p2[1])
    print(f"Transformed points: p1t={p1t}, p2t={p2t}")
    
    new_view = LineView.alloc().initWithPoints_([
        NSPoint(p1t[0], p1t[1]),
        NSPoint(p2t[0], p2t[1])
    ])
    overlay_window.setContentView_(new_view)
    line_view = new_view
    
    line_view.setNeedsDisplay_(True)
    overlay_window.display()
    overlay_window.orderFront_(None)
    print("Line updated successfully")

def clear_overlay():
    global overlay_window
    if overlay_window:
        overlay_window.orderOut_(None)
        overlay_window = None
        print("Overlay cleared")

def crop_board(image_path, top_left, bottom_right, output_path="ressources/cropped_board.png"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Impossible de charger l'image : {image_path}")
        return
    x1, y1 = top_left
    x2, y2 = bottom_right
    cropped_image = image[y1:y2, x1:x2]
    cv2.imwrite(output_path, cropped_image)
    #print(f"Image recadrée sauvegardée : {output_path}")

def preprocess_image(image):
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        gray = image  # L'image est déjà en niveaux de gris
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.equalizeHist(gray)
    return gray

# Nouvelle méthode de détection basée sur ORB
def detect_piece(square_image, piece_image, ratio_threshold=0.75, match_threshold=10):
    # Convertir en niveaux de gris et prétraiter
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

def remove_unwanted_colors(image, tolerance=20):  # Augmentation de la tolérance
    import numpy as np
    # Couleurs indésirables (en B, G, R)
    unwanted_colors = [
        np.array([90, 148, 122]),
        np.array([211, 236, 235]),
        np.array([140, 244, 248]),
        np.array([92, 204, 192])
    ]
    for col in unwanted_colors:
        lower = np.clip(col - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(col + tolerance, 0, 255).astype(np.uint8)
        # Debug : afficher la plage pour chaque couleur
        mask = cv2.inRange(image, lower, upper)
        image[mask != 0] = [0, 0, 0]
    return image

def split_and_identify_pieces(board_image_path, pieces_folder):
    board_image = cv2.imread(board_image_path)
    if board_image is None:
        print(f"Impossible de charger l'image : {board_image_path}")
        return

    board_image = remove_unwanted_colors(board_image)
    height, width = board_image.shape[:2]
    square_height = height // 8
    square_width = width // 8

    # Charger et redimensionner dynamiquement les pièces à la taille des cases
    pieces = {}
    for filename in os.listdir(pieces_folder):
        if filename.endswith(".png"):
            piece_name = os.path.splitext(filename)[0]
            piece_image = cv2.imread(os.path.join(pieces_folder, filename), cv2.IMREAD_UNCHANGED)
            
            # Vérifie la transparence
            if piece_image is None:
                continue
            if len(piece_image.shape) == 3 and piece_image.shape[2] == 4:
                piece_image = cv2.cvtColor(piece_image, cv2.COLOR_BGRA2BGR)

            # Redimensionne précisément à la taille des cases
            piece_resized = cv2.resize(piece_image, (square_width, square_height), interpolation=cv2.INTER_AREA)
            piece_gray = preprocess_image(piece_resized)
            pieces[piece_name] = piece_gray

    board_state = []
    probability_threshold = 0.35  # Réduction pour capturer plus de correspondances
    secondary_threshold = 0.2    # Seuil secondaire pour une vérification supplémentaire

    for row in range(8):
        board_row = []
        for col in range(8):
            y1, y2 = row * square_height, (row + 1) * square_height
            x1, x2 = col * square_width
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

            # Vérification secondaire pour capturer des correspondances faibles
            if best_prob < probability_threshold:
                for piece_name, piece_template in pieces.items():
                    res = cv2.matchTemplate(square_gray, piece_template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > secondary_threshold and max_val > best_prob:
                        best_prob = max_val
                        best_match = piece_name

            if best_prob < probability_threshold:
                best_match = "empty"
            board_row.append(best_match)
        board_state.append(board_row)

    # Affiche le résultat final
    for row in board_state:
        print(row)

    return board_state

def detect_piece_color(square_image):
    """
    Détecte la couleur dominante dans une case, en prenant une zone centrale (5x5),
    et en analysant la couleur moyenne en HSV.
    """
    height, width = square_image.shape[:2]
    center_y, center_x = int(height - height*0.2), width // 2
    offset = 2  # pour une zone 5x5
    # Correction de la découpe pour extraire une région 5x5 correctement
    region = square_image[center_y - offset:center_y + offset + 1, center_x - offset:center_x + offset + 1]
    region_hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(region_hsv.reshape(-1, 3), axis=0)
    h, s, v = avg_hsv
    if v > 150 and s < 50:
        return "B"  # pièce blanche
    elif v < 100:
        return "N"  # pièce noire
    else:
        return None  # incertain

def analyze_board_orb(cropped_board_path, pieces_folder):
    # Analyse un échiquier découpé en cases via detect_piece pour chaque case.
    board_image = cv2.imread(cropped_board_path)
    if board_image is None:
        print(f"Impossible de charger l'image : {cropped_board_path}")
        return
    board_image = remove_unwanted_colors(board_image)
    height, width = board_image.shape[:2]
    square_height = height // 8
    square_width = width // 8

    # Charger les templates des pièces
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

    probability_threshold = 0.5  # Ajusté pour être plus précis

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
                # Vérifier que les dimensions correspondent avant la comparaison
                if square_gray.shape != piece_template.shape:
                    piece_template = cv2.resize(piece_template, (square_gray.shape[1], square_gray.shape[0]))
                res = cv2.matchTemplate(square_gray, piece_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(res)
                if max_val > best_prob:
                    best_prob = max_val
                    best_match = piece_name

            # Appliquer le seuil pour éviter les faux positifs
            if best_prob < probability_threshold:
                best_match = "empty"
            else:
                # Détecter la couleur de la pièce si elle n'est pas vide
                piece_color = detect_piece_color(square)
                if piece_color:
                    best_match += piece_color

            board_row.append(best_match)
        board_state.append(board_row)

    #print("Board state from", cropped_board_path)
    #for row in board_state:
        #print(row)
    return board_state

def determine_active_color_from_bottom(board_state):
    white_row = -1
    black_row = -1
    for i, row in enumerate(board_state):
        for cell in row:
            if cell.startswith("roi"):
                if cell.endswith("B"):
                    white_row = i
                elif cell.endswith("N"):
                    black_row = i
    if white_row > black_row:
        return "w"
    else:
        return "b"

def board_state_to_fen(board_state, active_color):
    """
    Convertit l'état de l'échiquier en notation FEN.
    """
    piece_to_fen = {
        "roi": "k", "reine": "q", "tour": "r", "fou": "b", "cavalier": "n", "pion": "p"
    }
    fen_rows = []
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
                # Convertir les noms des pièces en notation FEN
                piece = square[:-1]  # Nom de la pièce
                color = square[-1]  # Couleur (B ou N)
                fen_piece = piece_to_fen.get(piece.lower(), "?")  # Récupérer le caractère FEN
                if color == "B":
                    fen_row += fen_piece.upper()  # Pièces blanches en majuscules
                else:
                    fen_row += fen_piece.lower()  # Pièces noires en minuscules
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    active_color = active_color  # Déterminer la couleur active
    fen = "/".join(fen_rows) + f" {active_color} - - 0 1"  # Ajout de la couleur active
    return fen

def get_best_move_from_stockfish(fen, stockfish_path="stockfish", max_retries=3):
    for i in range(max_retries):
        try:
            with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
                board = chess.Board(fen)
                result = engine.play(board, chess.engine.Limit(time=1.0))
                return result.move
        except chess.engine.EngineTerminatedError:
            print(f"Attempt {i+1}/{max_retries} failed: engine terminated unexpectedly")
    return None

def explain_move(move, board_state):
    """
    Explique le coup en langage explicite.
    """
    start_square = (8 - int(move[1]), ord(move[0]) - ord('a'))  # Convertir 'h1' en coordonnées (7, 7)
    end_square = (8 - int(move[3]), ord(move[2]) - ord('a'))    # Convertir 'g1' en coordonnées (7, 6)

    piece = board_state[start_square[0]][start_square[1]]  # Récupérer la pièce à déplacer
    piece_name = piece[:-1] if piece != "empty" else "aucune pièce"
    start_square_name = move[:2]
    end_square_name = move[2:]

    return f"Déplacez {piece_name} de {start_square_name} à {end_square_name}."

def play_best_move(move, board_top_left, screen_top_left, square_size):
    """
    Joue le meilleur coup en effectuant un glisser-déposer lent avec la pièce.
    Les coordonnées sont calculées par rapport au coin supérieur gauche de l'échiquier.
    """
    start_col = ord(move[0]) - ord('a')  # Convertir 'a'-'h' en 0-7
    start_row = 8 - int(move[1])         # Convertir '1'-'8' en 7-0
    end_col = ord(move[2]) - ord('a')
    end_row = 8 - int(move[3])

    # Calculer les coordonnées du milieu des cases
    start_x = board_top_left[0] + start_col * square_size + square_size // 2
    start_y = board_top_left[1] + start_row * square_size + square_size // 2
    end_x = board_top_left[0] + end_col * square_size + square_size // 2
    end_y = board_top_left[1] + end_row * square_size + square_size // 2

    # Ajouter les coordonnées du coin supérieur gauche de l'échiquier fourni
    start_x += screen_top_left[0]
    start_y += screen_top_left[1]
    end_x += screen_top_left[0]
    end_y += screen_top_left[1]

    # Effectuer un glisser-déposer lent
    pyautogui.moveTo(start_x, start_y, duration=0.5)  # Déplacer vers la pièce
    pyautogui.mouseDown()                            # Cliquer et maintenir
    pyautogui.moveTo(end_x, end_y, duration=1.0)     # Glisser vers la case cible
    pyautogui.mouseUp()                              # Relâcher le clic

def calculate_scale_factor():
    """
    Calcule le facteur d'échelle entre les dimensions de l'écran (pyautogui) et celles de l'image capturée (OpenCV).
    """
    # Dimensions de l'écran avec pyautogui
    screen_width, screen_height = pyautogui.size()

    # Capture d'écran avec pyautogui et dimensions avec OpenCV
    screenshot = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape

    # Calcul du facteur d'échelle
    scale_factor_x = image_width / screen_width
    scale_factor_y = image_height / screen_height

    # Vérification de la cohérence des facteurs d'échelle
    if abs(scale_factor_x - scale_factor_y) > 0.01:
        print("Attention : Les facteurs d'échelle X et Y diffèrent légèrement.")
    return scale_factor_x  # Retourne un facteur unique (supposant une échelle uniforme)

def verify_referential_consistency():
    """
    Vérifie si les dimensions de l'écran (pyautogui) et de l'image capturée (OpenCV) sont cohérentes.
    """
    # Dimensions de l'écran avec pyautogui
    screen_width, screen_height = pyautogui.size()
    #print(f"Dimensions de l'écran (pyautogui) : {screen_width}x{screen_height}")

    # Capture d'écran avec pyautogui et dimensions avec OpenCV
    screenshot = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    image_height, image_width, _ = image.shape
    #print(f"Dimensions de l'image capturée (OpenCV) : {image_width}x{image_height}")

    # Vérification de la cohérence
    if screen_width != image_width | screen_height != image_height:
        print("Attention : Les dimensions de l'écran et de l'image capturée ne correspondent pas.")
        print("Cela peut indiquer une différence de référentiel ou de mise à l'échelle.")
    else:
        print("Les dimensions de l'écran et de l'image capturée sont cohérentes.")

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
    win_bot_img = cv2.imread("ressources/win_bot.png")  # Ajout de l'image win_bot.png
    if board_img is None or (win_img is None and win_bot_img is None):
        return False

    # Vérification avec win.png
    if win_img is not None:
        res = cv2.matchTemplate(board_img, win_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return True

    # Vérification avec win_bot.png
    if win_bot_img is not None:
        res = cv2.matchTemplate(board_img, win_bot_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= threshold:
            return True

    return False

top_left = (477, 353)
bottom_right = (2077, 1954)

if __name__ == "__main__":
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

    # Lancer la boucle d'événements dans un thread séparé
    def run_event_loop():
        app = NSApplication.sharedApplication()
        app.run()

    event_thread = threading.Thread(target=run_event_loop, daemon=True)
    event_thread.start()

    # Attendre un peu pour que la boucle d'événements soit bien lancée
    time.sleep(1)

    # Calculer le facteur d'échelle
    scale_factor = calculate_scale_factor()

    # Initialiser l'overlay
    init_overlay_window()

    # Placer un point au milieu de l'écran
    screen_width, screen_height = pyautogui.size()
    center_x = screen_width // 2
    center_y = screen_height // 2
    print(f"Point au centre de l'écran avant mise à l'échelle: ({center_x}, {center_y})")
    
    # Appliquer le facteur d'échelle
    scaled_center_x = int(center_x / scale_factor)
    scaled_center_y = int(center_y / scale_factor)
    print(f"Point au centre de l'écran après mise à l'échelle: ({scaled_center_x}, {scaled_center_y})")
    create_point(scaled_center_x, scaled_center_y)

    previous_state = [
        ['tourN', 'cavalierN', 'fouN', 'reineN', 'roiN', 'fouN', 'empty', 'tourN'],
        ['pionN', 'pionN', 'pionN', 'empty', 'empty', 'pionN', 'pionN', 'pionN'],
        ['empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'empty', 'pionN', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'empty', 'pionB', 'pionN', 'empty', 'empty', 'empty'],
        ['empty', 'empty', 'cavalierB', 'empty', 'empty', 'empty', 'empty', 'empty'],
        ['pionB', 'pionB', 'pionB', 'empty', 'empty', 'pionB', 'pionB', 'pionB'],
        ['tourB', 'empty', 'fouB', 'reineB', 'roiB', 'empty', 'cavalierB', 'tourB']
    ]
    active_color = input("Entrez la couleur active (w pour blanc, b pour noir) : ").strip().lower()
    screenshot_path = "screenshot.png"
    subprocess.run(["screencapture", "-x", screenshot_path])
    crop_board(screenshot_path, top_left, bottom_right, "cropped_board.png")    
    state_A = analyze_board_orb("cropped_board.png", "pieces")

    # Si le joueur est blanc, jouer dès le départ
    if active_color == "w" and state_A:
        print("Joueur blanc - coup joué dès le départ")
        fen_A = board_state_to_fen(state_A, active_color)
        best_move_A = get_best_move_from_stockfish(fen_A, stockfish_path="stockfish")
        if best_move_A is not None:
            move_str = str(best_move_A)
            explanation = explain_move(move_str, state_A)
            print("Coup suggéré :", explanation)
            square_size = (bottom_right[0] - top_left[0]) // 8
            # Calculate the start and end points for the overlay
            start_x = top_left[0] + (ord(move_str[0]) - ord('a')) * square_size + square_size // 2
            start_y = top_left[1] + (8 - int(move_str[1])) * square_size + square_size // 2
            end_x = top_left[0] + (ord(move_str[2]) - ord('a')) * square_size + square_size // 2
            end_y = top_left[1] + (8 - int(move_str[3])) * square_size + square_size // 2

            print(f"Calculated coordinates before scaling:")
            print(f"Start: ({start_x}, {start_y})")
            print(f"End: ({end_x}, {end_y})")
            print(f"Scale factor: {scale_factor}")

            # Update the overlay with the move
            scaled_start_x = int(start_x / scale_factor)
            scaled_start_y = int(start_y / scale_factor)
            scaled_end_x = int(end_x / scale_factor)
            scaled_end_y = int(end_y / scale_factor)

            print(f"Scaled coordinates:")
            print(f"Start: ({scaled_start_x}, {scaled_start_y})")
            print(f"End: ({scaled_end_x}, {scaled_end_y})")

            update_line((scaled_start_x, scaled_start_y), (scaled_end_x, scaled_end_y))
            previous_state = update_board_state(state_A, move_str)
        print("Début de la partie...")

    while True:
        time.sleep(1)
        screenshot_path = "screenshot.png"
        subprocess.run(["screencapture", "-x", screenshot_path])
        crop_board(screenshot_path, top_left, bottom_right, "cropped_board.png")
        
        # Clear the overlay when the game is won
        if detect_win("cropped_board.png"):
            print("Partie Gagné ! Arrêt du script.")
            clear_overlay()
            sys.exit()
            
        state_A = analyze_board_orb("cropped_board.png", "pieces")
        
        if state_A:
            if previous_state is None:
                print("Etat initial de l'échiquier:")
                previous_state = state_A
            elif state_A != previous_state:
                diffs = diff_board_states(previous_state, state_A)
                for pos, prev, curr in diffs:
                    row, col = pos
                previous_state = state_A
                fen_A = board_state_to_fen(state_A, active_color)
                best_move_A = get_best_move_from_stockfish(fen_A, stockfish_path="stockfish")
                if best_move_A is None:
                    print("Aucun coup suggéré par Stockfish.")
                    continue
                move_str = str(best_move_A)
                explanation = explain_move(move_str, state_A)
                print("Coup suggéré :", explanation)
                square_size = (bottom_right[0] - top_left[0]) // 8
                # Calculate the start and end points for the overlay
                start_x = top_left[0] + (ord(move_str[0]) - ord('a')) * square_size + square_size // 2
                start_y = top_left[1] + (8 - int(move_str[1])) * square_size + square_size // 2
                end_x = top_left[0] + (ord(move_str[2]) - ord('a')) * square_size + square_size // 2
                end_y = top_left[1] + (8 - int(move_str[3])) * square_size + square_size // 2

                print(f"Calculated coordinates before scaling:")
                print(f"Start: ({start_x}, {start_y})")
                print(f"End: ({end_x}, {end_y})")
                print(f"Scale factor: {scale_factor}")

                # Update the overlay with the move
                scaled_start_x = int(start_x / scale_factor)
                scaled_start_y = int(start_y / scale_factor)
                scaled_end_x = int(end_x / scale_factor)
                scaled_end_y = int(end_y / scale_factor)

                print(f"Scaled coordinates:")
                print(f"Start: ({scaled_start_x}, {scaled_start_y})")
                print(f"End: ({scaled_end_x}, {scaled_end_y})")

                update_line((scaled_start_x, scaled_start_y), (scaled_end_x, scaled_end_y))
                previous_state = update_board_state(previous_state, move_str)
