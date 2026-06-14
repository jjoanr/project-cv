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
- El model és una **Prototypical Network** amb backbone ResNet-18 (`train_few_shot.py`). Pesos a la carpeta `model/`.
- Com que el dataset sintètic és molt similar a peces físiques reals estàndard, s'espera que el model funcioni bé sobre fotos de peces físiques reals estàndard **sense reentrenament**.

---

## Pipeline d'inferència (`inference.py`)

1. **Detecció del tauler**: l'usuari clica manualment les 4 cantonades en ordre **A1 → H1 → H8 → A8**
2. **Correcció de perspectiva**: warp del tauler a una imatge quadrada (orientació "a8_tl")
3. **Segmentació**: retallar les 64 caselles individuals
4. **Classificació few-shot**: per cada casella, trobar el prototip més proper
5. **Reconstrucció**: generar FEN + render digital del tauler

### Execució local

```bash
python inference.py --image Dataset/data/1856.jpg
```

El default de `--weights` apunta a `model/proto_resnet18_best.pth`.  
Per canviar el model: `--weights model/proto_resnet18_finetuned.pth`  
Per canviar el backbone: `--backbone resnet50`

### Models disponibles (`model/`)

| Fitxer | Descripció |
|--------|-----------|
| `proto_resnet18_best.pth` | Model principal — **millor rendiment** |
| `proto_resnet18_finetuned.pth` | Fine-tuning posterior — rendiment inferior |
| `proto_resnet18_minimal_data.pth` | Entrenament amb dades mínimes |

---

## Avaluació de prediccions (`eval_predictions.py`)

Script que compara els JSONs de `inference_output/` amb els ground truth de `Dataset/data/`.  
Prova automàticament les 4 rotacions (0°, 90° CW, 180°, 90° CCW) i tria la millor per cada tauler.

### Resultats sobre 6 taulers del dataset (1851–1856)

#### Model `proto_resnet18_best.pth` (MILLOR)

| Tauler | Rotació detectada | Encerts | Acc. | Peça ok/Color ≠ | Errors |
|--------|------------------|---------|------|-----------------|--------|
| 1851 | 0° | 64/64 | 100% | 0 | 0 |
| 1852 | 180° | 45/64 | 70.3% | 16 | 3 |
| 1853 | 90° CW | 60/64 | 93.8% | 4 | 0 |
| 1854 | 180° | 54/64 | 84.4% | 1 | 9 |
| 1855 | 0° | 62/64 | 96.9% | 2 | 0 |
| 1856 | 180° | 54/64 | 84.4% | 1 | 9 |
| **TOTAL** | — | **339/384** | **88.3%** | 24 | 21 |

#### Model `proto_resnet18_finetuned.pth` (PITJOR)

| Tauler | Rotació detectada | Encerts | Acc. |
|--------|------------------|---------|------|
| 1851 | 0° | 42/64 | 65.6% |
| 1852 | 180° | 29/64 | 45.3% |
| 1853 | 90° CW | 43/64 | 67.2% |
| 1854 | 180° | 46/64 | 71.9% |
| 1855 | 0° | 47/64 | 73.4% |
| 1856 | 180° | 31/64 | 48.4% |
| **TOTAL** | — | **238/384** | **62.0%** |

### Observacions sobre les rotacions

- **1851** i **1855**: orientació correcta (A1 clicat a baix-esquerra)
- **1852**, **1854**, **1856**: tauler rotat **180°** (A1 clicat a la cantonada oposada)
- **1853**: tauler rotat **90° CW**
- Per evitar errors d'orientació, assegurar-se de clicar **A1 primer** (cantonada inferior-esquerra des del punt de vista de les blanques)

### Tipus d'errors típics

- **Color incorrecte** (peça ok, _w vs _b): molt freqüent quan el tauler és rotat 180° — el model veu les peces blanques des del costat negre
- **Errors complets**: menys freqüents; principalment confusions entre peces similars (rook/queen, bishop/king)

---

## Pla de validació

### Fase 1 — Verificació del model (COMPLETADA)

- `inference.py` funciona amb selecció manual de cantonades
- Provat sobre imatges del Dataset A (1851–1856) — accuracy 88.3% amb `proto_resnet18_best.pth`
- Vegeu taula de resultats a la secció anterior

### Fase 2 — Few-shot amb peces noves (Dataset B)

Dues aproximacions a provar:

**A) Prototypical few-shot pur** (la idea original)
- Donar 5–10 imatges per classe de les noves peces
- El model calcula nous prototips sense reentrenar
- Proves fetes fins ara: accuracy no gaire alta, però es reportarà igualment

**B) Fine-tuning lleuger** (alternativa)
- Amb les 5–10 imatges de les noves peces, fer unes poques epochs d'entrenament addicional sobre el model existent
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
| `model/proto_resnet18_best.pth` | Fet — millors pesos |
| `model/proto_resnet18_finetuned.pth` | Fet — fine-tuning, rendiment inferior |
| `model/proto_resnet18_minimal_data.pth` | Fet — pesos amb dades mínimes |
| `inference.py` | Fet — selecció manual de cantonades per clic |
| `eval_predictions.py` | Fet — comparació prediccions vs ground truth amb detecció de rotació |
