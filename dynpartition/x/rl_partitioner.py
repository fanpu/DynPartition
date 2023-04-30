num_batches = 3
batch_size = 5  # 120
image_w = 128
image_h = 128


def train(model):
    model.train(True)

    for _ in range(num_batches):
        # generate random inputs
        inputs = torch.randn(batch_size, 3, image_w, image_h)

        # run forward pass
        outputs = model(inputs.to(DEVICE_0))


plt.switch_backend('Agg')

num_repeat = 10

stmt = "train(model)"


def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('ResNet50 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


partition_layer = 3


def single_gpu():
    setup = "import torchvision.models as models;" + \
            "model = models.resnet50(num_classes=num_classes).to(DEVICE_0)"
    sg_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    sg_mean, sg_std = np.mean(sg_run_times), np.std(sg_run_times)

    return sg_mean, sg_std


def model_parallelism():
    """Model paralellism without pipelining"""
    setup = f"model = ModelParallelResNet50(partition_layer={partition_layer})"
    mp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    mp_mean, mp_std = np.mean(mp_run_times), np.std(mp_run_times)

    return mp_mean, mp_std


def pipeline_parallelism():
    """Model parallelism with pipelining"""
    setup = f"model = PipelineParallelResNet50(partition_layer={partition_layer})"
    pp_run_times = timeit.repeat(
        stmt, setup, number=1, repeat=num_repeat, globals=globals())
    pp_mean, pp_std = np.mean(pp_run_times), np.std(pp_run_times)

    return pp_mean, pp_std


# sg_mean, sg_std = single_gpu()
# mp_mean, mp_std = model_parallelism()
# pp_mean, pp_std = pipeline_parallelism()

# plot([mp_mean, sg_mean, pp_mean],
#      [mp_std, sg_std, pp_std],
#      ['Model Parallel', 'Single GPU', 'Pipelining Model Parallel'],
#      get_plot_path().joinpath('mp_vs_rn_vs_pp.png'))
