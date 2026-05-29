# Projecte de Computer Vision — Few-Shot Chess Piece Recognition

**Autors:** Joan Rodríguez, Albert Casamitjana

## Descripció

Sistema de visió per computador centrat en **few-shot learning** per reconèixer peces d'escacs de qualsevol estil visual. El repte principal no és reconèixer peces estàndard (problema trivial amb CNNs modernes), sinó adaptar el sistema a un nou conjunt de peces desconegut amb tan sols **1–10 imatges per classe**.

El pipeline complet:
1. Detecta el tauler i corregeix la perspectiva (OpenCV)
2. Segmenta el tauler en 64 caselles individuals
3. Classifica cada casella mitjançant un model few-shot (prototypical network o Siamese network)
4. Reconstrueix la posició en notació FEN i genera una visualització digital

## Pipeline

| Fase | Descripció | Milestone |
|------|-----------|-----------|
| 1 | Detecció del tauler + correcció de perspectiva (OpenCV) | 50% |
| 2 | Segmentació en 64 caselles | 50% |
| 3 | Model few-shot: entrenament episòdic sobre peces estàndard | 50% |
| 4 | Avaluació N-shot: accuracy vs. nombre d'exemples (1/5/10-shot) | 100% |
| 5 | Reconstrucció FEN + render digital | 100% |

## Dataset

- **Dataset actual:** imatges sintètiques, ~400 MB, ubicat a `Dataset/data/`
- **Format:** parells `N.jpg` (imatge completa del tauler) + `N.json` per cada mostra
- **Estructura JSON:**
  ```json
  {
    "config": {"A3": "knight_b", "B4": "queen_w", ...},
    "corners": [[y1,x1], [y2,x2], [y3,x3], [y4,x4]]
  }
  ```
  - `config`: caselles ocupades amb etiqueta `{king,queen,rook,bishop,knight,pawn}_{w,b}`
  - `corners`: coordenades normalitzades dels 4 cantons del tauler — format `[y, x]` (no x,y)
- **Classes:** 12 classes ocupades + 1 classe `empty` = 13 classes total
- ~~ChessRender360 (Kaggle, 40 GB) — descartat per quota d'emmagatzematge al BSC~~

## Model

- **Arquitectura:** Prototypical Network (prioritat) o Siamese Network
- **Backbone:** ResNet-50 o EfficientNet com a feature extractor
- **Entrenament:** episòdic (N-way K-shot episodes) sobre peces estàndard
- **Inferència few-shot:** donat K exemples d'un nou estil, classifica les 13 classes sense re-entrenar
- **Mètrica principal:** accuracy en funció del K (1-shot, 5-shot, 10-shot) sobre peces no vistes durant l'entrenament

## Infraestructura

- Entrenament al **BSC** (Barcelona Supercomputing Center) — quota d'emmagatzematge limitada
- Entorn: contenidor **Singularity** existent, sense permisos per `pip install` fora del contenidor
- Treball local: Windows

## Fitxers existents

| Fitxer | Estat | Notes |
|--------|-------|-------|
| `train.py` | Obsolet | Enfocament YOLOv8m — antic, no alineat amb l'objectiu few-shot |
| `chess.yaml` | Obsolet | Config YOLO, 12 classes |
| `inspect_dataset.py` | Desactualitzat | Escrit per format COCO, no encaixa amb el JSON actual |
| `kaggle_train.ipynb` | No inspeccionat | Notebook d'entrenament |

## Objectiu central

Demostrar que el sistema pot reconèixer un nou estil de peces d'escacs **sense re-entrenament**, usant únicament 1–10 imatges de referència per classe. Comparar mètodes (prototypical vs. Siamese) i analitzar la corba accuracy vs. nombre d'exemples.
