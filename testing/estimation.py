import chess.pgn, os

def estimate_elo(pgn_file):
    """
    Estime l'ELO des deux joueurs à partir d'un fichier PGN.

    Args:
        pgn_file (str): Chemin vers le fichier PGN.

    Returns:
        dict: Un dictionnaire contenant les estimations d'ELO des deux joueurs.
    """
    with open(pgn_file, "r") as file:
        game = chess.pgn.read_game(file)

    if game is None:
        return {"error": "Aucune partie trouvée dans le fichier PGN."}

    white_elo = game.headers.get("WhiteElo", "Inconnu")
    black_elo = game.headers.get("BlackElo", "Inconnu")

    return {
        "White": white_elo,
        "Black": black_elo
    }

if __name__ == "__main__":
    pgn_path = os.path.join(os.getcwd(), "game.pgn")
    elo_estimations = estimate_elo(pgn_path)
    print("Estimations d'ELO :", elo_estimations)