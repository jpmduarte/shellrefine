# Comparação com métodos existentes (projeto survey)

Fonte dos valores de referência: `projeto_survey_fetal_head_segmentation_3d/results/benchmark_summary.csv`
(5-fold CV, bf16, mesmo dataset). Valores do nosso método: `experiment.md` / Estágio C
(2 seeds x 5 folds, pooled sobre os 3 finalistas, estatisticamente empatados entre si).

## Qualidade

| Modelo | Params (M) | Dice | HD95 (mm) | ASD (mm) |
|---|---|---|---|---|
| nnU-Net | 31.2 | 0.9466 ± 0.0041 | 3.457 | 1.057 |
| BiSegMamba | 57.7 | 0.9457 ± 0.0050 | 3.712 | 1.089 |
| **Boundary-first-then-refine (nosso)** | **2.80** | **0.9456 ± 0.0041** | **3.199** | **1.064** |
| MambaVesselNet++ | 151.4 | 0.9453 ± 0.0054 | 3.761 | 1.101 |
| SwinUNETR | 62.2 | 0.9370 ± 0.0141 | 4.238 | 1.266 |
| nnFormer | 37.6 | 0.9356 ± 0.0099 | 4.489 | 1.326 |
| TransBTS | 32.8 | 0.9342 ± 0.0237 | 4.475 | 1.713 |
| CoTr | 41.9 | 0.9308 ± 0.0178 | 4.896 | 1.387 |
| Elipsoide (baseline clássico) | — | 0.5031 ± 0.0745 | 25.350 | 13.492 |

*(ordenado por Dice; nnU-Net, BiSegMamba e o nosso método estão dentro de ~0.001 de
Dice um do outro — abaixo do próprio desvio-padrão de cada um, ou seja, empatados.)*

## Eficiência

| Modelo | Params (M) | MACs @128³ fwd (G) | VRAM treino (GB) | Latência/caso (s) |
|---|---|---|---|---|
| **Boundary-first-then-refine (nosso)** | **2.80** | **225.8** | **3.5** | **~1.03 †** |
| nnFormer | 37.6 | 62.5 | 12.44 | 7.115 |
| nnU-Net | 31.2 | 477.2 | 19.26 | 4.281 |
| CoTr | 41.9 | 665.1 | 28.97 | 4.753 |
| BiSegMamba | 57.7 | 388.7 | 36.31 | 5.915 |
| MambaVesselNet++ | 151.4 | 1441.6 | 45.41 | 8.565 |
| SwinUNETR | 62.2 | 762.8 | 54.73 | 7.452 |
| TransBTS | 32.8 | 322.6 | 55.17 | 5.615 |
| Elipsoide (baseline clássico) | — | — | — | 29.242 |

*(ordenado por params; MACs medidos com `torch.utils.flop_counter.FlopCounterMode`
num input sintético de 128³, o mesmo protocolo/patch de referência do survey, para
ficar diretamente comparável — não é o patch real usado no nosso pipeline, que é
muito mais pequeno (32³ na Fase 1, tiles de 64³ na Fase 2). VRAM treino = pico entre
as duas fases (Fase 1: 0.7GB, Fase 2: 3.5GB). † latência nossa medida em fp32, sem
warmup nem repetições; a do survey é bf16 com warmup + 3 repetições — não é 100%
comparável, valor preliminar.)*

## Qualidade por parâmetro (Dice / milhão de parâmetros)

| Modelo | Dice | Params (M) | Dice/M params |
|---|---|---|---|
| **Boundary-first-then-refine (nosso)** | 0.9456 | **2.80** | **0.3377** |
| nnFormer | 0.9356 | 37.6 | 0.0249 |
| nnU-Net | 0.9466 | 31.2 | 0.0303 |
| TransBTS | 0.9342 | 32.8 | 0.0285 |
| CoTr | 0.9308 | 41.9 | 0.0222 |
| BiSegMamba | 0.9457 | 57.7 | 0.0164 |
| SwinUNETR | 0.9370 | 62.2 | 0.0151 |
| MambaVesselNet++ | 0.9453 | 151.4 | 0.0062 |
