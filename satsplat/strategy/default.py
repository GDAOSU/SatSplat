import gsplat.strategy


class DefaultStrategy(gsplat.strategy.DefaultStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
