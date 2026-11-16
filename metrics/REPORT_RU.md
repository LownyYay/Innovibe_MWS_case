# A_official_train_val_r32 — метрики сохранённого запуска

Dev и официальный val входили в обучение temporal-модели. Метрики только для диагностики seen-val.
Статус запуска: `complete`.
Время по фазам — подготовка 0.46 мин, smoke 0.66 мин, обучение 31.62 мин, инференс dev 7.18 мин, инференс public 8.51 мин.
dev: сгенерировано 373/373, пропуски по времени 0, ошибки выполнения 0.
public: сгенерировано 383/383, пропуски по времени 0, ошибки выполнения 0.

Шаги обучения: 1551/1551. Шаг выбранного адаптера: 1551.
Dev: интервал 1050/1455; начало 1068/1455; конец 1350/1455; invalid 0/373.

| Ветка | Temporal до → после | Fixed / broken eligible | Fixed / broken все matched | State до → после |
|---|---|---|---|---|
| v3_history | 324/679 → 376/679 (0.4772 → 0.5538) | 72/20 | 95/36 | 352/623 → 360/623 |
| v3_no_history | 308/679 → 376/679 (0.4536 → 0.5538) | 80/12 | 103/28 | 352/623 → 360/623 |
| agreement_no_history | 312/679 → 376/679 (0.4595 → 0.5538) | 78/14 | 103/30 | 352/623 → 360/623 |

Все official after: P=0.5922; R=0.6146; F1=0.60319149; state=0.5778; grounding_turn=0.8236; grounding_span=0.9626.
n_calls=383; n_gold=1845; n_pred=1915; matched=1134; n_temporal=679; n_state=623; n_grounded=1134.
Public parse errors: 18/383.

Все from/end/full, eligibility, matched/unmatched, state fixed/broken и проверки matching/полей — summary.json и */evaluation/.
Для воспроизведения используйте исходный RESULT ZIP и оценщик с закреплённым MANIFEST.json.
Source RESULT SHA256: eb1f980b3fad1eb811a69320fecbbcc5ae135da419b4987284920939d6f4bee0
