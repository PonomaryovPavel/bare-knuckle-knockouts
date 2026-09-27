# Как получена выгрузка

Файлы в `data_raw` — ответы API английской Википедии, по одному на год:

```bash
for y in 2018 2019 2020 2021 2022 2023 2024 2025 2026; do
  curl -sS -G "https://en.wikipedia.org/w/api.php" \
    --data-urlencode "action=parse" \
    --data-urlencode "page=${y} in Bare Knuckle Fighting Championship" \
    --data-urlencode "prop=wikitext" \
    --data-urlencode "format=json" \
    --data-urlencode "formatversion=2" \
    -A "ваш-контакт" \
    -o "data_raw/bkfc_${y}.json"
  sleep 1
done
```

Выгрузка датирована 27 сентября 2026 года. Статьи правятся, поэтому повторное
скачивание даст другие числа; для воспроизведения расчётов используйте файлы
из репозитория.
