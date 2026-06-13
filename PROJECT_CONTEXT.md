# Context del Projecte — Few-Shot Chess Piece Recognition

## Objectiu principal

Donat una foto d'un tauler d'escacs físic, generar la representació digital completa:
detecció del tauler → segmentació per caselles → classificació de peces → render virtual + FEN.

El sistema ha de funcionar no només per peces estàndard, sinó per **qualsevol estil de peces** (formes especials, disseny propi, etc.) usant **few-shot learning**: amb tan sols 5–10 imatges de referència per classe, el model ha de reconèixer les noves peces **sense reentrenar**.

---

## Dataset i entrenament

- **Dataset A (sintètic)**: fotos reals d'un tauler físic amb peces estàndard, posicions generades aleatòriament. Cada imatge té un JSON amb les coordenades de les 4 cantonades del tauler i la posició de cada peça. ~400 MB a `Dataset/data/`.
- **13 classes**: king_w, queen_w, rook_w, bishop_w, knight_w, pawn_w, king_b, queen_b, rook_b, bishop_b, knight_b, pawn_b, empty.
- El pipeline d'entrenament usava el JSON per detectar i segmentar les 64 caselles (`extract_piece_crops.py` + `refine_graella.py`).
- El model és una **Prototypical Network** amb backbone ResNet-18 (`train_few_shot.py`). Pesos: `proto_resnet18_best.pth`.
- Com que el dataset sintètic és molt similar a peces físiques reals estàndard, s'espera que el model funcioni bé sobre fotos de peces físiques reals estàndard **sense reentrenament**.

---

## Pipeline d'inferència (`inference.py`)

1. **Detecció del tauler**: trobar les 4 cantonades a la foto
2. **Correcció de perspectiva**: warp del tauler a una imatge quadrada
3. **Segmentació**: retallar les 64 caselles individuals
4. **Classificació few-shot**: per cada casella, trobar el prototip més proper
5. **Reconstrucció**: generar FEN + render digital del tauler

---

## Problemes actuals i solucions proposades

### Problema 1: Detecció de cantonades sense JSON

Per a les fotos d'inferència (Dataset A no usat en entrenament, peces físiques reals, Dataset B), **no hi ha JSON**. La detecció automàtica per visió per computador (Hough lines, contorns) no és prou robusta per a fons complexos (rajoles, taules, etc.).

**Solució acordada**: quan l'usuari executa el programa, el programa obre la imatge i demana que l'usuari **cliqui manualment les 4 cantonades** del tauler.

### Problema 2: Orientació del tauler

Sense saber l'angle de la foto, no es pot determinar quina cantonada és A1, H1, A8, H8.

**Solució acordada**: l'usuari clica les 4 cantonades **en ordre específic**: A1 → H1 → H8 → A8 (o similar, a definir). Això elimina l'ambigüitat d'orientació.

---

## Pla de validació (per ordre de prioritat)

### Fase 1 — Verificació del model (PRIORITAT ACTUAL)

1. Fer que `inference.py` funcioni amb selecció manual de cantonades (clic de l'usuari)
2. Provar sobre imatges del Dataset A **no usades en entrenament** — hauria d'encertar quasi tot
3. Provar sobre fotos reals de peces estàndard físiques — s'espera bona accuracy sense reentrenament

### Fase 2 — Few-shot amb peces noves (Dataset B)

Dues aproximacions a provar:

**A) Prototypical few-shot pur** (la idea original)
- Donar 5–10 imatges per classe de les noves peces
- El model calcula nous prototips sense reentrenar
- Proves fetes fins ara: accuracy no gaire alta, però es reportarà igualment

**B) Fine-tuning lleuger** (alternativa)
- Amb les 5–10 imatges de les noves peces, fer unes poques epochs d'entrenament addicional sobre el model existent
- Proves fetes: bona accuracy
- Terminologia correcta: sí, això és **fine-tuning**

### Com proporcionar les imatges de referència per Dataset B

Dos mètodes proposats:

**Mètode 1 — Una peça per foto**
Posar només una peça al tauler, fer la foto. El sistema detecta automàticament quina casella té la peça (les altres estan buides). Repetir per cada classe.

**Mètode 2 — Posició inicial**
Fer 5–10 fotos del tauler en posició inicial de partida (la posició estàndard). Com que la posició inicial és coneguda, el sistema sap automàticament quina peça va a cada casella, sense necessitat d'anotar manualment.

---

## Estat dels fitxers

| Fitxer | Estat |
|--------|-------|
| `extract_piece_crops.py` | Fet — segmenta i organitza per classe |
| `refine_graella.py` | Fet — corregeix perspectiva a partir de cantonades |
| `train_few_shot.py` | Fet — entrenament episòdic Prototypical Network |
| `proto_resnet18_best.pth` | Fet — pesos del model entrenat |
| `inference.py` | Parcialment fet — funciona amb JSON, **falta selecció manual de cantonades** |

---

## Propera tasca immediata

Implementar a `inference.py` la **selecció manual de cantonades per clic**:
- Obrir la foto en una finestra OpenCV
- L'usuari clica les 4 cantonades en ordre (A1, H1, H8, A8)
- El programa continua amb el pipeline normal
- Això elimina la necessitat de JSON i resol l'orientació alhora
