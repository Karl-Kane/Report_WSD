"""WSD vs cosine vs Schedule-Free на CIFAR-10 (GPU).

Вопрос эксперимента: что делать, если длина обучения T заранее неизвестна.
Для каждого бюджета B (эпох) сравниваются пять стратегий:
  1. cosine(B)          — cosine, настроенный ровно на B: отдельное обучение на каждый бюджет;
  2. cosine(Tmax)→B     — cosine на максимальный бюджет, остановленный на эпохе B;
  3. const              — постоянный lr без затухания, веса на эпохе B;
  4. WSD                — один прогон с постоянным lr + короткое затухание (20% бюджета) из сохранённой точки;
  5. Schedule-Free SGD  — один прогон без расписания, оценка на эпохе B.

Запуск:
    python wsd_experiment.py                  # полный эксперимент (~1 час на RTX 4060)
    python wsd_experiment.py --quick          # проверочный прогон (~3 мин)
    python wsd_experiment.py --plot-only      # перерисовать графики из results/results.json
"""
import argparse, copy, json, math, os, time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results')

p = argparse.ArgumentParser()
p.add_argument('--budgets', type=int, nargs='+', default=[10, 20, 40])
p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
p.add_argument('--lr', type=float, default=0.2, help='пиковый lr для cosine/WSD/const')
p.add_argument('--sf-lr', type=float, default=1.0, help='lr для Schedule-Free')
p.add_argument('--batch', type=int, default=512)
p.add_argument('--decay-frac', type=float, default=0.2, help='доля бюджета на затухание WSD')
p.add_argument('--warmup-epochs', type=float, default=2.0)
p.add_argument('--quick', action='store_true')
p.add_argument('--plot-only', action='store_true')
args = p.parse_args()
if args.quick:
    args.budgets, args.seeds = [2, 4], [0]

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------- данные (целиком на GPU)
def load_cifar():
    import torchvision
    root = os.path.join(HERE, 'data')
    tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
    te = torchvision.datasets.CIFAR10(root, train=False, download=True)
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=dev).view(1, 3, 1, 1)
    prep = lambda x: ((torch.tensor(x, device=dev).permute(0, 3, 1, 2).float() / 255) - mean) / std
    return (prep(tr.data), torch.tensor(tr.targets, device=dev),
            prep(te.data), torch.tensor(te.targets, device=dev))


def augment(x, gen):
    """Случайный сдвиг (crop с padding 4) и отражение по горизонтали — на GPU."""
    n = x.shape[0]
    flip = torch.rand(n, device=dev, generator=gen) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    xp = F.pad(x, (4, 4, 4, 4), mode='reflect')
    dx, dy = torch.randint(0, 9, (2, n), device=dev, generator=gen)      # свой сдвиг у каждой картинки
    i = torch.arange(32, device=dev)
    rows, cols = (dy[:, None] + i)[:, :, None], (dx[:, None] + i)[:, None, :]
    return xp.permute(0, 2, 3, 1)[torch.arange(n, device=dev)[:, None, None], rows, cols].permute(0, 3, 1, 2)


def batches(X, Y, seed, epoch, bs):
    """Порядок и аугментация зависят только от (seed, epoch):
    продолжение с сохранённой точки видит ровно те же батчи, что и непрерывный прогон."""
    gen = torch.Generator(device=dev).manual_seed(seed * 100003 + epoch)
    perm = torch.randperm(X.shape[0], device=dev, generator=gen)
    for k in range(0, X.shape[0] - bs + 1, bs):          # неполный последний батч отбрасываем
        idx = perm[k:k + bs]
        yield augment(X[idx], gen), Y[idx]


# ---------------------------------------------------------------- модель: маленькая ResNet
def conv_bn(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = conv_bn(c, c), conv_bn(c, c)

    def forward(self, x):
        return x + self.b(self.a(x))


def make_model():
    return nn.Sequential(
        conv_bn(3, 64),
        conv_bn(64, 128), nn.MaxPool2d(2), Res(128),
        conv_bn(128, 256), nn.MaxPool2d(2),
        conv_bn(256, 512), nn.MaxPool2d(2), Res(512),
        nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(512, 10),
    ).to(dev, memory_format=torch.channels_last)


# ---------------------------------------------------------------- расписания (lr на шаге t)
def make_schedules(steps_per_epoch):
    W = int(args.warmup_epochs * steps_per_epoch)        # разогрев одинаковой длины у всех

    def warm(t):
        return args.lr * (t + 1) / W

    def cosine(T):
        return lambda t: warm(t) if t < W else args.lr * 0.5 * (1 + math.cos(math.pi * (t - W) / (T - W)))

    def const(t):
        return warm(t) if t < W else args.lr

    def wsd(T):
        T_dec = int(args.decay_frac * T)
        return lambda t: warm(t) if t < W else (args.lr if t < T - T_dec else args.lr * (T - t) / T_dec)

    return cosine, const, wsd


# ---------------------------------------------------------------- обучение / оценка
@torch.no_grad()
def evaluate(model, X, Y):
    model.eval()
    correct, loss = 0, 0.0
    for k in range(0, X.shape[0], 1000):
        with torch.autocast(dev, dtype=torch.bfloat16):
            out = model(X[k:k + 1000].contiguous(memory_format=torch.channels_last)).float()
        loss += F.cross_entropy(out, Y[k:k + 1000], reduction='sum').item()
        correct += (out.argmax(1) == Y[k:k + 1000]).sum().item()
    model.train()
    return correct / X.shape[0], loss / X.shape[0]


def run(seed, lr_fn, epochs, state=None, start_epoch=0, eval_at=(), save_at=(), log=None):
    """Обучение обычным SGD+momentum. state — (model, opt) для продолжения с сохранённой точки.
    eval_at — эпохи, после которых считать test acc; save_at — эпохи, после которых сохранить веса."""
    if state is None:
        torch.manual_seed(seed)
        model = make_model()
        opt = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9, nesterov=True, weight_decay=5e-4)
    else:
        model, opt = state
    steps = Xtr.shape[0] // args.batch
    evals, saved = {}, {}
    for ep in range(start_epoch, start_epoch + epochs):
        for k, (xb, yb) in enumerate(batches(Xtr, Ytr, seed, ep, args.batch)):
            t = ep * steps + k
            for g in opt.param_groups:
                g['lr'] = lr_fn(t)
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb.contiguous(memory_format=torch.channels_last)), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if log is not None:
            log.append({'epoch': ep + 1, 'lr': lr_fn((ep + 1) * steps - 1), 'test_acc': evaluate(model, Xte, Yte)[0]})
        if ep + 1 in eval_at:
            evals[ep + 1] = evaluate(model, Xte, Yte)[0]
        if ep + 1 in save_at:
            saved[ep + 1] = (copy.deepcopy(model), copy.deepcopy(opt))
    return evals, saved, (model, opt)


def fix_saved_opt(model, opt):
    """deepcopy(opt) ссылается на параметры исходной модели — перепривязываем к копии."""
    new = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9, nesterov=True, weight_decay=5e-4)
    new.load_state_dict(opt.state_dict())
    return new


def run_schedule_free(seed, epochs, eval_at, log=None):
    import schedulefree
    torch.manual_seed(seed)
    model = make_model()
    steps = Xtr.shape[0] // args.batch
    opt = schedulefree.SGDScheduleFree(model.parameters(), lr=args.sf_lr, momentum=0.9,
                                       weight_decay=5e-4, warmup_steps=int(args.warmup_epochs * steps))
    evals = {}
    for ep in range(epochs):
        model.train(); opt.train()
        for xb, yb in batches(Xtr, Ytr, seed, ep, args.batch):
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xb.contiguous(memory_format=torch.channels_last)), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if ep + 1 in eval_at or log is not None:
            # оценка на усреднённых весах x; статистики BatchNorm пересчитываем под них
            opt.eval(); model.train()
            with torch.no_grad():
                for i, (xb, _) in enumerate(batches(Xtr, Ytr, seed, 10_000 + ep, args.batch)):
                    with torch.autocast(dev, dtype=torch.bfloat16):
                        model(xb.contiguous(memory_format=torch.channels_last))
                    if i == 30:
                        break
            acc = evaluate(model, Xte, Yte)[0]
            if ep + 1 in eval_at:
                evals[ep + 1] = acc
            if log is not None:
                log.append({'epoch': ep + 1, 'lr': args.sf_lr, 'test_acc': acc})
    return evals


# ---------------------------------------------------------------- эксперимент
def experiment():
    B, TM = sorted(args.budgets), max(args.budgets)
    steps = Xtr.shape[0] // args.batch
    cosine, const, wsd = make_schedules(steps)
    starts = {b: min(int(round((1 - args.decay_frac) * b)), b - 1) for b in B}   # с какой эпохи начинается затухание
    res = {k: {str(b): [] for b in B} for k in ['cosine(B)', 'cosine(Tmax)→B', 'const', 'WSD', 'Schedule-Free']}
    curves = {}
    t0 = time.time()
    for seed in args.seeds:
        # 1, 2. cosine, настроенный на каждый бюджет; самый длинный заодно «останавливаем» на меньших B
        for b in B:
            log = [] if b == TM else None
            ev, _, _ = run(seed, cosine(b * steps), b, eval_at=set(B) | {b}, log=log)
            res['cosine(B)'][str(b)].append(ev[b])
            if b == TM:
                for bb in B:
                    res['cosine(Tmax)→B'][str(bb)].append(ev[bb])
                curves.setdefault('cosine(Tmax)', []).append(log)
            print(f'[seed {seed}] cosine({b}) = {ev[b]:.4f}   ({time.time() - t0:.0f} c)', flush=True)

        # 3, 4. один основной прогон с постоянным lr: const на эпохах B, сохранения для WSD
        log = []
        ev, saved, _ = run(seed, const, TM, eval_at=set(B), save_at=set(starts.values()), log=log)
        curves.setdefault('const', []).append(log)
        for b in B:
            res['const'][str(b)].append(ev[b])
        for b in B:
            model, opt = saved[starts[b]]
            model = copy.deepcopy(model)
            opt = fix_saved_opt(model, opt)
            blog = [] if b == TM else None
            ev, _, _ = run(seed, wsd(b * steps), b - starts[b], state=(model, opt), start_epoch=starts[b],
                           eval_at={b}, log=blog)
            res['WSD'][str(b)].append(ev[b])
            if blog is not None:
                curves.setdefault('WSD decay', []).append(blog)
            print(f'[seed {seed}] const@{b} = {res["const"][str(b)][-1]:.4f}   WSD({b}) = {ev[b]:.4f}', flush=True)

        # 5. Schedule-Free: один прогон, оценки на эпохах B
        log = []
        ev = run_schedule_free(seed, TM, set(B), log=log)
        curves.setdefault('Schedule-Free', []).append(log)
        for b in B:
            res['Schedule-Free'][str(b)].append(ev[b])
        print(f'[seed {seed}] Schedule-Free: ' + ', '.join(f'{b}: {ev[b]:.4f}' for b in B), flush=True)

    compute = {'cosine(B)': sum(B), 'WSD': starts[TM] + sum(b - starts[b] for b in B), 'Schedule-Free': TM}
    out = {'args': vars(args), 'results': res, 'curves': curves, 'compute_epochs': compute,
           'minutes': (time.time() - t0) / 60, 'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else 'cpu'}
    with open(os.path.join(OUT, 'results.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    return out


# ---------------------------------------------------------------- графики и таблица
COLORS = {'cosine(B)': '#1f77b4', 'cosine(Tmax)→B': '#9ecae1', 'const': '#7f7f7f',
          'WSD': '#ff7f0e', 'Schedule-Free': '#2ca02c'}
LABELS = {'cosine(B)': 'cosine, настроенный на B (отдельное обучение)',
          'cosine(Tmax)→B': 'cosine на max бюджет, остановлен на B',
          'const': 'постоянный lr', 'WSD': 'WSD (один прогон + затухание)', 'Schedule-Free': 'Schedule-Free'}


def report(out):
    res, B = out['results'], sorted(int(b) for b in out['results']['WSD'])
    lines = ['| стратегия | ' + ' | '.join(f'{b} эпох' for b in B) + ' |', '|---|' + '---|' * len(B)]
    for k, v in res.items():
        cells = [f'{np.mean(v[str(b)]) * 100:.2f} ± {np.std(v[str(b)]) * 100:.2f}' for b in B]
        lines.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    c = out['compute_epochs']
    lines += ['', f'Вычислений на все бюджеты: cosine(B) — {c["cosine(B)"]} эпох, WSD — {c["WSD"]} эпох, '
                  f'Schedule-Free — {c["Schedule-Free"]} эпох.', f'GPU: {out["gpu"]}, время: {out["minutes"]:.1f} мин.']
    text = '\n'.join(lines)
    print('\n' + text)
    with open(os.path.join(OUT, 'results.md'), 'w', encoding='utf-8') as f:
        f.write('# Test accuracy, %, CIFAR-10 (среднее ± std по seed)\n\n' + text + '\n')

    # 1. точность при разных бюджетах
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, v in res.items():
        m = [np.mean(v[str(b)]) * 100 for b in B]
        s = [np.std(v[str(b)]) * 100 for b in B]
        ax.errorbar(B, m, yerr=s, marker='o', capsize=4, lw=2.5 if k in ('WSD', 'cosine(B)') else 1.5,
                    color=COLORS[k], label=LABELS[k])
    ax.set(xlabel='бюджет обучения, эпох', ylabel='test accuracy, %', title='CIFAR-10: качество при разных бюджетах')
    ax.set_xticks(B); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'budgets.png'), dpi=150); plt.close(fig)

    # 2. кривые точности по эпохам: видно скачок на затухании WSD
    cv = out['curves']
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    for k, color in [('cosine(Tmax)', COLORS['cosine(B)']), ('const', COLORS['const']),
                     ('Schedule-Free', COLORS['Schedule-Free'])]:
        e = [r['epoch'] for r in cv[k][0]]
        a = np.mean([[r['test_acc'] * 100 for r in run_] for run_ in cv[k]], 0)
        ax[0].plot(e, a, color=color, label={'cosine(Tmax)': f'cosine на {max(B)} эпох'}.get(k, LABELS.get(k, k)))
        if k != 'Schedule-Free':                  # у Schedule-Free lr другого масштаба и смысла
            ax[1].plot(e, [r['lr'] for r in cv[k][0]], color=color,
                       label={'cosine(Tmax)': f'cosine на {max(B)} эпох'}.get(k, LABELS.get(k, k)))
    d = cv['WSD decay']
    e = [r['epoch'] for r in d[0]]
    ax[0].plot(e, np.mean([[r['test_acc'] * 100 for r in run_] for run_ in d], 0), color=COLORS['WSD'], lw=2.5,
               label='WSD: затухание')
    ax[1].plot(e, [r['lr'] for r in d[0]], color=COLORS['WSD'], lw=2.5, label='WSD: затухание')
    ax[0].set(xlabel='эпоха', ylabel='test accuracy, %', title='Точность по ходу обучения', ylim=(60, None))
    ax[1].set(xlabel='эпоха', ylabel='lr', title='Расписания (WSD = постоянный lr + затухание)')
    for a_ in ax:
        a_.grid(alpha=0.3)
        a_.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'curves.png'), dpi=150); plt.close(fig)

    # 3. стоимость
    fig, ax = plt.subplots(figsize=(6, 4))
    ks = ['cosine(B)', 'WSD', 'Schedule-Free']
    ax.bar([LABELS[k].split(' (')[0].split(',')[0] for k in ks], [c[k] for k in ks], color=[COLORS[k] for k in ks])
    for i, k in enumerate(ks):
        ax.text(i, c[k], str(c[k]), ha='center', va='bottom', fontsize=12)
    ax.set(ylabel='эпох обучения', title=f'Стоимость моделей на бюджеты {", ".join(map(str, B))}')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'compute.png'), dpi=150); plt.close(fig)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if args.plot_only:
        with open(os.path.join(OUT, 'results.json'), encoding='utf-8') as f:
            report(json.load(f))
    else:
        print('Устройство:', torch.cuda.get_device_name(0) if dev == 'cuda' else 'CPU (будет очень долго!)')
        Xtr, Ytr, Xte, Yte = load_cifar()
        report(experiment())
