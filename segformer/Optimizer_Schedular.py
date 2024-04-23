

class PeakScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self, optimizer,
            epoch_size=-1,
            lr_start=0.000001,
            lr_max=0.000005 * Config.BATCH_SIZE,
            lr_min=0.000001,
            lr_ramp_ep=4,
            lr_sus_ep=0,
            lr_decay=0.8,
            verbose=True
    ):
        self.epoch_size = epoch_size
        self.optimizer = optimizer
        self.lr_start = lr_start
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_ramp_ep = lr_ramp_ep
        self.lr_sus_ep = lr_sus_ep
        self.lr_decay = lr_decay
        self.is_plotting = True

        epochs = list(range(Config.EPOCHS))
        learning_rates = []
        for i in epochs:
            self.epoch = i
            learning_rates.append(self.get_lr())
        self.is_plotting = False
        self.epoch = 0
        plt.scatter(epochs, learning_rates)
        plt.show()
        super(PeakScheduler, self).__init__(optimizer, verbose=verbose)

    def get_lr(self):
        if not self.is_plotting:
            if self.epoch_size == -1:
                self.epoch = self._step_count - 1
            else:
                self.epoch = (self._step_count - 1) / self.epoch_size

        if self.epoch < self.lr_ramp_ep:
            lr = (self.lr_max - self.lr_start) / self.lr_ramp_ep * self.epoch + self.lr_start

        elif self.epoch < self.lr_ramp_ep + self.lr_sus_ep:
            lr = self.lr_max
        else:
            lr = (self.lr_max - self.lr_min) * self.lr_decay ** (
                        self.epoch - self.lr_ramp_ep - self.lr_sus_ep) + self.lr_min
        return [lr for _ in self.optimizer.param_groups]