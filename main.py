import cv2
import numpy as np
import os
import time
import subprocess
import chess
import chess.engine
import pyautogui
import random
import sys  # nouvel import

top_left = (478, 353)
bottom_right = (2078, 1957)

top_left_B = (1756, 366)
bottom_right_B = (3308, 1917)

def crop_board(image_path, top_left, bottom_right, output_path="cropped_board.png"):
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

def perform_drag_and_drop(move, board_top_left, square_size):
    """
    Effectue un glisser-déposer avec la souris pour déplacer une pièce.
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

    # Effectuer un glisser-déposer lent
    pyautogui.moveTo(start_x, start_y, duration=0.5)  # Déplacer vers la pièce
    pyautogui.mouseDown()                            # Cliquer et maintenir
    pyautogui.moveTo(end_x, end_y, duration=1.0)     # Glisser vers la case cible
    pyautogui.mouseUp()                              # Relâcher le clic

def perform_drag_and_drop_with_pyautogui(move, board_top_left, square_size, scale_factor):
    """
    Effectue un glisser-déposer avec la souris pour déplacer une pièce en utilisant pyautogui.
    Les coordonnées sont ajustées avec un facteur d'échelle pour correspondre à la résolution de l'écran.
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

    # Appliquer le facteur d'échelle pour ajuster les coordonnées
    start_x = int(start_x / scale_factor)
    start_y = int(start_y / scale_factor)
    end_x = int(end_x / scale_factor)
    end_y = int(end_y / scale_factor)

    # Effectuer un glisser-déposer lent (messages de debug supprimés)
    pyautogui.moveTo(start_x, start_y, duration=0.1)
    pyautogui.mouseDown()
    pyautogui.moveTo(end_x, end_y, duration=0.1)
    pyautogui.mouseUp()

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

def select_active_color():
    try:
        result = subprocess.check_output(
            ["osascript", "-e", 'display dialog "Entrez la couleur active (w pour blanc, b pour noir) :" default answer ""']
        )
        result = result.decode("utf-8")
        for line in result.splitlines():
            if "text returned:" in line:
                return line.split("text returned:", 1)[1].strip().lower() or "w"
        return input("Entrez la couleur active (w pour blanc, b pour noir) : ").strip().lower() or "w"
    except Exception as e:
        print("Erreur lors de l'affichage de la boîte de dialogue:", e)
        return input("Entrez la couleur active (w pour blanc, b pour noir) : ").strip().lower() or "w"

def detect_win(cropped_board_path, win_image_path="win.png", threshold=0.8):
    board_img = cv2.imread(cropped_board_path)
    win_img = cv2.imread(win_image_path)
    if board_img is None or win_img is None:
        return False
    res = cv2.matchTemplate(board_img, win_img, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return max_val >= threshold

if __name__ == "__main__":
    # Calculer le facteur d'échelle
    scale_factor = calculate_scale_factor()
    #print(f"Facteur d'échelle calculé : {scale_factor}")

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
    # Déterminer la couleur active à partir de l'état initial
    time.sleep(3)
    screenshot_path = "screenshot.png"
    subprocess.run(["screencapture", "-x", screenshot_path])
    crop_board(screenshot_path, top_left, bottom_right, "cropped_board_A.png")    
    state_A = analyze_board_orb("cropped_board_A.png", "pieces")
    active_color = select_active_color()  # Remplacement de l'input par une fenêtre de dialogue
    time.sleep(3)
    
    # Si le joueur est blanc, jouer dès le départ
    if active_color == "w" and state_A:
        print("Joueur blanc - coup joué dès le départ")
        fen_A = board_state_to_fen(state_A, active_color)
        #print("FEN initial :", fen_A)
        best_move_A = get_best_move_from_stockfish(fen_A, stockfish_path="stockfish")
        if best_move_A is not None:
            move_str = str(best_move_A)
            explanation = explain_move(move_str, state_A)
            print("Coup joué :", explanation)
            square_size = (bottom_right[0] - top_left[0]) // 8
            perform_drag_and_drop_with_pyautogui(move_str, top_left, square_size, calculate_scale_factor())
            previous_state = update_board_state(state_A, move_str)
            print("Nouvel état de l'échiquier mis à jour.")
        print("Début de la partie...")
    
    while True:
        time.sleep(random.uniform(0.5, 15))
        screenshot_path = "screenshot.png"
        subprocess.run(["screencapture", "-x", screenshot_path])
        crop_board(screenshot_path, top_left, bottom_right, "cropped_board_A.png")
        
        # Arrêter le script si win.png est détectée sur l'échiquier
        if detect_win("cropped_board_A.png"):
            print("Image win.png détectée. Arrêt du script.")
            sys.exit()
            
        state_A = analyze_board_orb("cropped_board_A.png", "pieces")
        
        if state_A:
            if previous_state is None:
                print("Etat initial de l'échiquier:")
                previous_state = state_A
            elif state_A != previous_state:
                diffs = diff_board_states(previous_state, state_A)
                print("Modification détectée dans l'échiquier:")
                for pos, prev, curr in diffs:
                    row, col = pos
                    print(f"  Case ({row},{col}) : {prev} -> {curr}")
                previous_state = state_A
                fen_A = board_state_to_fen(state_A, active_color)
                #print("FEN mis à jour :", fen_A)
                best_move_A = get_best_move_from_stockfish(fen_A, stockfish_path="stockfish")
                if best_move_A is None:
                    print("Aucun coup suggéré par Stockfish.")
                    continue
                move_str = str(best_move_A)
                explanation = explain_move(move_str, state_A)
                print("Coup joué :", explanation)
                square_size = (bottom_right[0] - top_left[0]) // 8
                perform_drag_and_drop_with_pyautogui(move_str, top_left, square_size, calculate_scale_factor())
                previous_state = update_board_state(previous_state, move_str)
                print("Nouvel état de l'échiquier mis à jour. Partie en cours...")
