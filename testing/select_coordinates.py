import cv2, time
import pyautogui
import numpy as np

def select_coordinates():
    """
    Capture l'écran total et permet de sélectionner des coordonnées en cliquant.
    """
    screenshot = pyautogui.screenshot()
    image = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    coordinates = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coordinates.append((x, y))
            print(f"Coordonnée sélectionnée : ({x}, {y})")
            # Dessiner un point sur l'image
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Sélectionnez les coordonnées", image)

    # Afficher l'image et capturer les clics
    cv2.imshow("Sélectionnez les coordonnées", image)
    cv2.setMouseCallback("Sélectionnez les coordonnées", click_event)
    print("Cliquez sur les coins de l'échiquier (coin supérieur gauche et coin inférieur droit).")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(coordinates) < 2:
        print("Vous devez sélectionner au moins deux coordonnées.")
        return None

    return coordinates

if __name__ == "__main__":
    time.sleep(2)  # Temps pour passer à l'écran de jeu
    coords = select_coordinates()
    if coords:
        print(f"Coordonnées sélectionnées : {coords}")
