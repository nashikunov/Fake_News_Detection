# Применение глубокого обучения к задаче распознавания фейковых новостей

* Курсовая работа НИУ ВШЭ БИ 2020. Москва
* Шикунов Николай
* Научный руководитель: Ковалев Евгений

* В работе используется датасет **LIAR-PLUS**
* Оригинальный датасет [LIAR](https://github.com/thiagorainmaker77/liar_dataset) содержит 12.836 коротких новостных заголовков, взятых из ресурса [POLITIFACT](https://www.politifact.com/). Данные размечены на предмет фейка экспертами из POLITIFACT. 

**Таргет**
* **pants-fire** - Выдуманная новость 
* **false** - Новость ложная
* **barely-true** - В целом новость ложная  
* **half-true** - Новость наполовину правдивая
* **mostly-true** - В целом новость правдивая
* **true** - Правдивая новость

**6 таргетов можно представить в виде бинарной классификации**
* **true_news** = 1, если новость не фейк (half-true или mostly-true или true == 1)
* **true_news** = 0, если новость фейк (pants-fire или false или barely-true == 1)

**Признаки датасета LIAR**
* **Column 1**: (**id_json**) ID заголовка ([ID].json);
* **Column 2**: (**label**) Target label;
* **Column 3**: (**statement**) Новостной заголовок;
* **Column 4**: (**subject**) Тема новостного заголовка (Здравоохранение, выборы, налоги и тд);
* **Column 5**: (**speaker**) Имя спикера новостного заголовка (Трамп, Клинтон и тд);
* **Column 6**: (**speaker_job**) Профессия спикера новостного заголовка (Президент, губернатор и тд);
* **Column 7**: (**state**) Штат(город), где появился новостной заголовок (Нью-Йорк, Техас и тд);
* **Column 8**: (**party**) Политическая партия спикера новостного заголовка (Демократ, республикант);
* **Columns 9-13**: Суммарная история новостей, сказанных спикером (Сумма по всем Target label). Включая текущий новостной заголовок;
* **9**: Сумма (**barely_true_counts**) 'barely true' новостей спикера;
* **10**: Сумма (**false_counts**) 'false' новостей спикера;
* **11**: Сумма (**half_true_counts**) 'half true' новостей спикера;
* **12**: Сумма (**mostly_true_counts**) 'mostly true' новостей спикера;
* **13**: Сумма (**pants_on_fire_counts**) 'pants on fire' новостей спикера;
* **Column 14**: (**context**) Контекст новости. Место, где появилась данная новость (Интервью, пресс-релиз, Твиттер и тд);

* Датасет [LIAR-PLUS](https://github.com/Tariq60/LIAR-PLUS) является расширением датасета [LIAR](https://github.com/thiagorainmaker77/liar_dataset). Добавился новый признак **justification**

* **Column 15**: (**justification**) Обоснование новости, которое написали эксперты в процессе факт-чекинга;

* ["Liar, Liar Pants on Fire": A New Benchmark Dataset for Fake News Detection](https://arxiv.org/pdf/1705.00648.pdf)
* [LIAR dataset](https://github.com/thiagorainmaker77/liar_dataset)
* [LIAR-PLUS dataset](https://github.com/Tariq60/LIAR-PLUS)
