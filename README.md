# bachelor-thesis

## Разработка архитектуры глубокого обучения способной обрабатывать табличные данные для решения задач мета-обучения

Код разработанной архитектуры содержится в папке `deepsets`

`MetaLearning-GAN` содержит некоторые мои эксперименты с LM-GAN

Инструкции по запуску:
1. Скачиваем [processed_data.zip](https://disk.yandex.ru/d/QnxlQAvZJs3WNA) -- train и test датасеты
2. Содержимое архива необходимо поместить в `MetaLearning-GAN/meta_gan/processed_data`и в `deepsets/processed_data`
3. В `deepsets` для запуска удобнее всего пользоваться ноутбуками `deepset_approach.ipynb` и `deepset_gan_approach.ipynb`
4. В `MetaLearning-GAN/meta_gan` запуск производится через `Trainer.py`


В этом репозитории присутствует код из следующих репозиториев:
1. [MetaLearning-GAN](https://github.com/IlyaHalsky/MetaLearning-GAN)
2. [deepsets_digitsum](https://github.com/dpernes/deepsets-digitsum)
