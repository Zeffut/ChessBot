import chess
import chess.engine

def test_stockfish(stockfish_path="stockfish"):
    try:
        # Initialiser le moteur Stockfish
        with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
            print("Stockfish démarré avec succès.")

            # Exemple de position FEN (position initiale des échecs)
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            board = chess.Board(fen)
            print("Position FEN :")
            print(board)

            # Calculer le meilleur coup
            result = engine.play(board, chess.engine.Limit(time=1.0))
            print(f"Meilleur coup suggéré par Stockfish : {result.move}")

    except FileNotFoundError:
        print(f"Erreur : Stockfish introuvable à l'emplacement '{stockfish_path}'.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    test_stockfish(stockfish_path="stockfish")
