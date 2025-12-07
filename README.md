# Gesture Recognition

Projeto simples de reconhecimento de gestos usando OpenCV e scikit-learn.

Estrutura:

```
gesture-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── dataset/                # CSVs gerados na coleta
│
├── src/
│   ├── collect_data.py     # Script de captura de gestos
│   ├── train_model.py      # Treinamento do modelo ML
│   ├── real_time.py        # Reconhecimento em tempo real
│   └── utils.py            # Funções auxiliares
│
└── models/
    └── gesture_model.pkl   # Modelo treinado (gerado pelo treinamento)
```

Rápido guia de uso:

- Instale dependências:

```bash
python3 -m pip install -r requirements.txt
```

- Coletar exemplos: rode `src/collect_data.py`. Use as teclas numéricas para rotular as amostras.

- Treinar modelo: rode `src/train_model.py` (gera `models/gesture_model.pkl`).

- Rodar reconhecimento em tempo real: `src/real_time.py`.

Observações:
- Os scripts usam uma ROI central do frame para capturar o gesto (ajustável).
- O formato do dataset é CSV com coluna `label` seguida pelos pixels redimensionados e normalizados.

Licença: MIT (livre para uso e estudo).

# -Sistema-de-Reconhecimento-de-Gestos-
# 🤖 Sistema de Reconhecimento de Gestos com Python + MediaPipe + Machine Learning  Este projeto implementa um sistema completo de **reconhecimento de gestos das mãos em tempo real** usando:  - Python - OpenCV - MediaPipe Hands - Scikit-Learn  O sistema captura landmarks da mão, treina um modelo de Machine Learning.
