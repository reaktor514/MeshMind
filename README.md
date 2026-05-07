# MeshMind

**MeshMind** — своя нейросеть для 2D/3D моделирования. Генерирует картинки и
3D-меши по текстовому промпту и/или картинке-референсу.

```text
text  ─┐                          ┌─►  PNG  (Stable Diffusion Turbo)
       ├─► MeshMindPipeline  ────►┤
image ─┘                          └─►  .obj / .glb  (ShapE + MeshMindRefiner)
```

## Что внутри

```
meshmind/
├── config.py            — общая конфигурация (env-переменные MESHMIND_*)
├── pipeline.py          — оркестрация всех пайплайнов
├── models/
│   ├── text_to_image.py — обёртка над Stable Diffusion Turbo (text→image)
│   ├── image_to_image.py— обёртка над SD img2img (reference→image)
│   ├── text_to_mesh.py  — обёртка над ShapE (text→mesh)
│   ├── image_to_mesh.py — обёртка над ShapE-img2img (image→mesh)
│   └── refiner.py       — MeshMindRefiner: своя архитектура (transformer)
└── utils/
    ├── image_io.py      — load_image / save_image
    ├── mesh_io.py       — экспорт mesh в .obj / .glb / .ply / .stl
    └── seeding.py       — детерминированный seeding всех RNG
app.py                   — Gradio UI (2 вкладки: 2D / 3D)
```

### MeshMindRefiner — собственная архитектура

`meshmind.models.MeshMindRefiner` — это компактный трансформер с
self-attention + cross-attention к текстовому эмбеддингу, написанный с нуля.
Применяется к латентам после VAE / перед декодером ShapE для повышения
качества. Веса инициализированы так, что без обучения модуль ведёт себя как
identity (residual passthrough), поэтому пайплайн работает «из коробки», а
после файнтюна — улучшает результат.

Ключевые свойства:

* SwiGLU FFN, RMS-стабильное self-attention.
* Cross-attention к CLIP text embeddings (768-dim по умолчанию).
* Поддерживает оба формата латентов: 4D `(B, C, H, W)` (image) и 3D
  `(B, N, C)` (token).
* Параметризация — `dim`, `depth`, `heads` — через `MeshMindConfig`.

## Установка

Нужен Python 3.10+. Для GPU-инференса — CUDA-сборка PyTorch.

```bash
git clone https://github.com/reaktor514/MeshMind.git
cd MeshMind
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Или как пакет (для установки зависимостей и `meshmind-app` CLI):

```bash
pip install -e .
```

## Запуск

### Веб-интерфейс (Gradio)

```bash
python app.py
```

Откроется на `http://localhost:7860`. Две вкладки:

* **2D генерация** — text → image, опционально с reference image (img2img).
* **3D генерация** — text → mesh или reference image → mesh, экспорт в
  `.glb` / `.obj` / `.ply` / `.stl`.

### Python API

```python
from meshmind import MeshMindPipeline

pipe = MeshMindPipeline.from_pretrained()

# 2D
images = pipe.text_to_image("a red cyber-dragon, studio render", num_images=2)
images[0].save("dragon.png")

# 3D
mesh = pipe.text_to_mesh("a stylized low-poly fox", seed=42)
mesh.export("fox.glb")

# 3D от референса
from PIL import Image
ref = Image.open("photo.jpg")
mesh = pipe.image_to_mesh(ref)
mesh.export("from_photo.obj")
```

## Конфигурация

Все параметры можно переопределить env-переменными:

| Переменная | Дефолт | Описание |
|---|---|---|
| `MESHMIND_T2I_MODEL` | `stabilityai/sd-turbo` | Веса для text→image |
| `MESHMIND_I2I_MODEL` | `stabilityai/sd-turbo` | Веса для image→image |
| `MESHMIND_T2M_MODEL` | `openai/shap-e` | Веса для text→mesh |
| `MESHMIND_I2M_MODEL` | `openai/shap-e-img2img` | Веса для image→mesh |
| `MESHMIND_IMAGE_SIZE` | `512` | Размер 2D-изображений |
| `MESHMIND_IMAGE_STEPS` | `4` | Шаги диффузии для 2D |
| `MESHMIND_MESH_STEPS` | `64` | Шаги диффузии для 3D |
| `MESHMIND_MESH_RESOLUTION` | `128` | Разрешение grid для marching cubes |
| `MESHMIND_REFINER_DIM` | `256` | Скрытая размерность MeshMindRefiner |
| `MESHMIND_REFINER_DEPTH` | `4` | Глубина MeshMindRefiner |
| `MESHMIND_REFINER_STRENGTH` | `0.0` | Сила residual-добавки рефайнера (0 = identity) |
| `MESHMIND_DEVICE` | `auto` | `cuda`, `cpu` или `auto` |
| `MESHMIND_DTYPE` | `auto` | `float32`, `float16`, `bfloat16` или `auto` |

## Системные требования

* **2D** (SD-Turbo) — ~6 GB VRAM, шустро на любой современной GPU; на CPU ~30 c.
* **3D** (ShapE) — ~6 GB VRAM, ~10–30 с на GPU, ~3–5 мин на CPU.

## Лицензия

[MIT](LICENSE). Веса бэкбонов распространяются под их собственными лицензиями
(SD-Turbo — Stability AI Community License, ShapE — Apache-2.0).
