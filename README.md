# CompactFactExtraction A · README по воспроизведению

Двухстадийный экстрактор фактов из русскоязычных обращений в техническую поддержку. В выбранном варианте A экстрактор с LoRA r16 выделяет факты и временные выражения. Детерминированный слой нормализует даты, восстанавливает границы интервалов и согласует переходы состояний. Вторая Qwen3-0.6B с LoRA r32 уточняет даты уже извлечённых фактов. Проверка схемы и защитные правила сохраняют прежние факты при невалидном ответе temporal-модели.

Temporal-адаптер A обучался на official train + official val. Общий dev также входил в обучение. Результат 0,5538 на official val показывает качество на виденных данных. Скрытый тест и усиленный ASR-срез не оценивались. Допуск обучения на official val в доступных материалах кейса не подтверждён.

## Состав архива

| Содержимое | Путь |
| --- | --- |
| Веса экстрактора r16 и temporal-адаптера A r32 | `weights/extractor/` и `weights/temporal_A/` |
| Docker, HTTP-инференс, календарный repair и benchmark | `Dockerfile` и `inference/` |
| Исходные данные v2.1 и адаптер тёплого старта экстрактора | `training/extractor_inputs/` |
| Данные, код, конфигурация и оценщик temporal A | `training/QWEN06_A_official_train_val_r32.zip` |
| Архивные прогнозы, официальный оценщик и проверки | `metrics/` |
| График и таблица точек | `resources/resources_quality.png` и `resources/quality.csv` |
| Презентация и запрос, проверенный в Docker | `Defense.pptx` и `demo/request_verified_cpu.json` |

В Linux проверьте целостность архива после распаковки. Две контрольные суммы обученных весов и область оценки также указаны в [artifact_identity.json](artifact_identity.json).

```sh
sha256sum -c MANIFEST.sha256
```

## Результат

| Метрика v2.1 | Базовый GLM-5.3 | Вариант A, seen-val |
|---|---:|---:|
| Fact precision | 0,3348 | 0,5922 |
| Fact recall | 0,3805 | 0,6146 |
| Fact F1 | 0,3562 | 0,6032 |
| Temporal recall | 0,4706, 208/442 | 0,5538, 376/679 |
| State accuracy | 0,5345, 333/623 | 0,5778, 360/623 |
| Grounding turn | 0,7835 | 0,8236 |
| Grounding span | — | 0,9626 |

На всех 383 звонках совпало 1 134 из 1 915 предсказанных фактов при 1 845 эталонных. У разных экстракторов различается множество сопоставленных фактов, по которым считается temporal recall. По опубликованной конфигурации GLM-5.3 (744 млрд) в ансамбле A **в 608,65 раза меньше параметров**. Это подтверждает ресурсный порог по числу параметров. Оценка качества A относится к виденным звонкам. [График ресурсов и качества](resources/resources_quality.png) · [Методика](resources/BENCHMARK.md).

## Пересчёт опубликованных метрик

Достаточно Python 3.11+ и стандартной библиотеки. Из корня проекта пересчитайте baseline и вариант A одним официальным scorer v2.1. Прогнозы A уже приложены.

```sh
python -B metrics/scorer/case_metrics.py metrics/scorer/val.jsonl training/extractor_inputs/official/baseline_pred.jsonl
python -B metrics/scorer/case_metrics.py metrics/scorer/val.jsonl metrics/predictions_seen_val.jsonl --json reproduced_a.json
python -c "import json; assert json.load(open('reproduced_a.json')) == json.load(open('metrics/reproduced_official_score.json')); print('A: metrics match')"
```

Ожидаются 383 звонка, temporal recall 0,5538 и state accuracy 0,5778. Повторный подсчёт даёт те же числа на сохранённых ответах. Свежая генерация temporal A использует sampling и может дать другие даты. После проверки HTTP-схемой из архивного выхода удалён один факт с несовместимой полярностью. Отдельный пересчёт этого выхода приведён в [integration_checks.json](metrics/integration_checks.json).

## Запуск сервиса

### Docker на CPU

Команды выполняются в корне проекта. Образ включает обе LoRA и базу Qwen3-0.6B на ревизии `c1899de289a04d12100db370d81485cdf75e47ca`. База загружается при сборке. После сборки сервис работает с локальными весами.

```sh
docker build -t compactfact:a .
docker run --rm -p 127.0.0.1:8000:8080 compactfact:a --device cpu
```

Для запуска готового образа в Ubuntu 24.04 в WSL из PowerShell:

```powershell
wsl -d Ubuntu-24.04 -u root -- docker run --rm -p 127.0.0.1:8000:8080 compactfact:a --device cpu
```

### Docker на одной NVIDIA GPU

```sh
docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu128 -t compactfact:a-cu128 .
docker run --rm --gpus 1 -p 127.0.0.1:8000:8080 compactfact:a-cu128 --device cuda
```

### HTTP-запрос

```sh
curl --fail http://127.0.0.1:8000/readyz
curl --fail --include --max-time 180 -H "Content-Type: application/json" --data-binary "@demo/request_verified_cpu.json" http://127.0.0.1:8000/v1/extract
```

В PowerShell используйте `curl.exe`. Сервис принимает `call_id`, `started_at` и `turns` и возвращает `{call_id, facts}`. Совместимые пути `/health` и `/extract` также доступны. Входные метки и дополнительные поля реплик исключаются из промптов. HTTP-слой сверяет спикера, оффсеты и цитаты с исходной репликой, удаляет неверные необязательные поля и сообщает число исправлений в `X-Evidence-Issue-Count`. Режимы `--mode span`, `repair` и `full` доступны в одном образе. По умолчанию включён полный каскад A без истории. Temporal-этап использует sampling, поэтому новый результат может отличаться от записанного `demo/recorded_response_seen_val.json`.

На CPU в WSL проверен полный каскад A. Реальный запрос вернул HTTP 200 и четыре факта, некорректный запрос получил HTTP 422. [Результат проверки](metrics/docker_smoke.json).

Для видео запустите контейнер в отдельном окне. Во втором покажите готовность сервиса, отправьте запрос и выделите в ответе `facts`, даты и `source.turn_idx`. В Windows 11 сочетание **Win + Shift + R** открывает запись области экрана. Ожидание ответа на CPU можно вырезать из записи. [Инструкция Microsoft](https://support.microsoft.com/en-us/windows/apps/use-snipping-tool-to-capture-screenshots).

## Повторное обучение

### Экстрактор r16

Запускайте в Linux с Python 3.12 и CUDA PyTorch. Исходный запуск использовал PyTorch 2.11.0+cu128 и одну GPU. Команды используют официальный набор v2.1 и проверенный адаптер тёплого старта. Оба входят в архив.

```sh
python3.12 -m venv .venv-extractor
source .venv-extractor/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -B inference/extractor/download_model.py --output base_model
python -B inference/extractor/train.py --data training/extractor_inputs/official --model base_model --output runs/extractor-preflight --preflight
python -B inference/extractor/train.py --data training/extractor_inputs/official --model base_model --init-adapter training/extractor_inputs/init_run01 --output runs/extractor-refit --seed 42 --epochs 2 --batch 1 --accum 8 --max-length 12288 --save-steps 100
python -B inference/extractor/evaluate.py --data training/extractor_inputs/official --model base_model --adapter runs/extractor-refit/adapter --out runs/extractor-eval
deactivate
```

SFT использует LoRA rank 16 и learning rate 2e-4. Чекпойнт экстрактора выбирается по loss на official val. Выходные веса находятся в `runs/extractor-refit/adapter/`. [Манифест исходного запуска](training/extractor_inputs/training_manifest.json).

### Temporal-адаптер A r32

В архиве temporal A есть 4 139 обучающих заданий из 4 309 звонков, зафиксированные задания dev/public, код и CPU-оценщик. В dev/public повторно используются фиксированные входы, согласованные с поставленным экстрактором. Для обучения рассчитана одна RTX 5090 с BF16 и Python 3.12. Seed 42, три эпохи, LoRA r32/α64, learning rate 2e-4, batch 1, accumulation 8. Сохраняется адаптер последней эпохи.

На Clore распакуйте worker ZIP, откройте `Train_A.ipynb` в папке с `trial_config.json`, задайте свободный `RUN_NAME` и запустите **Run All**. Эквивалентные команды из корня комплекта:

```sh
mkdir -p runs/temporal-retrain
unzip -q training/QWEN06_A_official_train_val_r32.zip -d runs/temporal-retrain
cd runs/temporal-retrain/A_official_train_val_r32
python3.12 -m venv .venv312
.venv312/bin/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
.venv312/bin/python -m pip install -r code/requirements.txt
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false .venv312/bin/python -B code/worker.py --root "$PWD" --run-name replay01 --mode train
unzip -q CPU_EVALUATOR.zip -d cpu-eval
.venv312/bin/python -B cpu-eval/qwen06_evaluator/code/score_one.py --coordinator cpu-eval/qwen06_evaluator --result-zip A_official_train_val_r32_replay01_RESULT.zip --out A_replay01_METRICS
```

Режим `--mode check` проверяет токенизацию, `--mode smoke` добавляет короткую CUDA-проверку. Полный режим `train` запускает обе проверки перед обучением. Прогон сохраняет `*_RESULT.zip` с весами и генерациями. CPU-оценщик записывает новый отчёт и предсказания. Новый результат следует сравнить с исходным отдельно. Сервис проверяет SHA-256 поставленных весов, поэтому для развёртывания нового результата потребуется обновить `artifact_identity.json` и приложить его собственные метрики.

## Замеры и ограничения

Для замера режима `full` внутри Linux-образа передайте публичный JSONL. Команду выполняйте из корня проекта.

```sh
docker run --rm --entrypoint python -e OMP_NUM_THREADS=6 -e MKL_NUM_THREADS=6 --mount "type=bind,source=$PWD/resources,target=/calls,readonly" compactfact:a -B inference/benchmark.py /calls/benchmark_smoke_calls.jsonl --device cpu --mode full
```

Тем же набором измерьте `--mode span` и `--mode repair`. Для режима CUDA используйте образ `compactfact:a-cu128`, пробросьте `--gpus 1` до имени образа и передайте `--device cuda` бенчмарку. Этот пример проверяет протокол на малом наборе. Для отчётных значений используйте одну достаточно большую выборку для всех точек. Сохраняйте оборудование, режим, длительности, RSS/VRAM и пропускную способность. [Методика и график](resources/BENCHMARK.md).

Проверка этой сборки включает целостность весов, пересчёт 383 сохранённых звонков, совпадение 383 выходов детерминированного repair и один живой запрос на CPU. Новый полный цикл обучения, CUDA-инференс и сквозная ресурсная серия в этой сборке не запускались. [Полный отчёт](REPORT.md).
