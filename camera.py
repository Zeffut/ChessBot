import cv2
import numpy as np
import os
import chess.engine
import sys

def detect_chessboard(image):
    """
    Détecte l'emplacement de l'échiquier dans une image.
    
    Args:
        image (numpy.ndarray): Image à analyser
        
    Returns:
        tuple: ((x1, y1), (x2, y2)) où:
               (x1, y1) sont les coordonnées du coin supérieur gauche
               (x2, y2) sont les coordonnées du coin inférieur droit
               None si aucun échiquier n'est détecté
    """
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
    for match in raw_matches:
        if len(match) >= 2:  # Vérifiez qu'il y a au moins deux correspondances
            m, n = match
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    return len(good_matches) > match_threshold

# Ouvrir la webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

# Ajoutez une fonction pour afficher le meilleur coup
def display_best_move_with_arrow(frame, move_str, square_width):
    start_col = ord(move_str[0]) - ord('a')
    start_row = 8 - int(move_str[1])
    end_col = ord(move_str[2]) - ord('a')
    end_row = 8 - int(move_str[3])
    
    # Calculer les positions de départ et d'arrivée
    start_x = start_col * square_width + square_width // 2
    start_y = start_row * square_width + square_width // 2
    end_x = end_col * square_width + square_width // 2
    end_y = end_row * square_width + square_width // 2
    
    # Tracer la flèche
    cv2.arrowedLine(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2, tipLength=0.1)

def remove_unwanted_colors(image, tolerance=30):
    unwanted_colors = [
        np.array([90, 148, 122]),
        np.array([211, 236, 235]),
        np.array([140, 244, 248]),
        np.array([92, 204, 192]),
        np.array([68, 204, 192]),
        np.array([80, 169, 119]),
        np.array([68, 255, 254]),
        np.array([66, 137, 96]),
        np.array([167, 198, 193])
    ]
    for col in unwanted_colors:
        lower = np.clip(col - tolerance, 0, 255).astype(np.uint8)
        upper = np.clip(col + tolerance, 0, 255).astype(np.uint8)
        mask = cv2.inRange(image, lower, upper)
        image[mask != 0] = [0, 0, 0]
    return image

def detect_piece_color(square_image):
    height, width = square_image.shape[:2]
    
    margin = int(min(height, width) * 0.2)
    center_region = square_image[margin:height - margin, margin:width - margin]
    
    region_hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
    
    avg_h, avg_s, avg_v = np.mean(region_hsv.reshape(-1, 3), axis=0)

    #print(f"[DEBUG] H: {avg_h:.1f} | S: {avg_s:.1f} | V: {avg_v:.1f}")
    
    # Adapter les seuils :
    if avg_v > 65:
        return "B"  # Pièce blanche
    elif avg_v <= 65:
        return "N"  # Pièce noire
    else:
        return "incertain"



def analyze_board_orb(board_image, pieces_folder):
    global debug_counter
    if board_image is None:
        print("Impossible de charger l'image.")
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
                if piece_color == "incertain":
                    print(f"Couleur incertaine pour la case ({row}, {col})")
                    best_match += "?"  # Indiquer une incertitude
                elif piece_color:
                    best_match += piece_color
            board_row.append(best_match)
        board_state.append(board_row)
    return board_state


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
            elif square.endswith("?"):  # Gérer les pièces incertaines
                print(f"Pièce incertaine détectée: {square}")
                continue  # Ignorer les pièces incertaines
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
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    # Créer la chaîne FEN
    fen = "/".join(fen_rows) + f" {active_color} - - 0 1"
    return fen

def get_best_move_from_stockfish(fen, stockfish_path='stockfish', max_retries=3):
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

def display_piece_names(frame, board_state, position=(10, 50), font_scale=0.5, color=(255, 255, 255)):
    square_height, square_width = frame.shape[0] // 8, frame.shape[1] // 8
    for row in range(8):
        for col in range(8):
            piece = board_state[row][col]
            if piece != "empty":
                piece_name = piece[:-1]  # Enlever la couleur pour afficher juste le nom
                text_position = (col * square_width + 5, row * square_width + 15)
                cv2.putText(frame, piece_name, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire le flux vidéo.")
        break

    # Détecter l'échiquier dans l'image
    chessboard_coords = detect_chessboard(frame)

    if chessboard_coords is not None:
        # Découper l'échiquier
        (x1, y1), (x2, y2) = chessboard_coords
        chessboard_image = frame[y1:y2, x1:x2].copy()  # Conserver une copie de l'image originale
        
        # Analyser l'état du plateau
        state_A = analyze_board_orb(chessboard_image, "ressources/pieces")

        if state_A:
            active_color = detect_active_color(state_A)

            try:
                # Générer la chaîne FEN
                fen_A = board_state_to_fen(state_A, active_color)

                # Si un meilleur coup est trouvé, afficher le coup
                best_move_A = get_best_move_from_stockfish(fen_A)  # Obtenir le meilleur coup
                if best_move_A is not None:
                    move_str = str(best_move_A)
                    display_best_move_with_arrow(chessboard_image, move_str, chessboard_image.shape[1] // 8)  # Afficher la flèche sur l'overlay

            except ValueError as e:
                print(f"Erreur lors de la génération de la chaîne FEN : {e}")

        # Afficher l'image découpée de l'échiquier sans traitement
        cv2.imshow('Échiquier détecté', chessboard_image)  # Afficher l'image originale avec l'overlay

    else:
        # Si aucun échiquier n'est détecté, afficher une image noire
        cv2.imshow('Échiquier détecté', np.zeros_like(frame))  # Affiche une image noire

    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
