Теория искусственных нейронных сетей (МГТУ им. Баумана, ИУ9)

расписания lr — WSD и Schedule-Free
- `report_WSD/Доклад_WSD_ScheduleFree.ipynb` — эксперимент на MNIST (как в РК1)
- `report_WSD/wsd_experiment.py` — эксперимент на CIFAR-10 (PyTorch, GPU)
- `report_WSD/results/` — таблица и графики
- `report_WSD/presentation/Доклад_WSD.pptx` — презентация

Запуск эксперимента на CIFAR-10 (~1 час на RTX 4060):

    pip install torch torchvision schedulefree matplotlib
    python report_WSD/wsd_experiment.py            # полный прогон
    python report_WSD/wsd_experiment.py --quick    # быстрая проверка