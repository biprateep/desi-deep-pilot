import numpy as np
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import AutoLocator, NullFormatter, NullLocator, ScalarFormatter


class PowerLawScale(mscale.ScaleBase):
    """Custom class defining a Power law scaler for the axes"""

    name = "power_law"

    def __init__(self, axis, *, gamma, **kwargs):
        super().__init__(axis)
        self.gamma = gamma

    def set_default_locators_and_formatters(self, axis):
        """
        Default
        """
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())

    def limit_range_for_scale(self, vmin, vmax, minpos):

        return vmin, vmax

    def get_transform(self):
        """Set the actual transform for the axis coordinates."""
        return self.PowerLawTransform(self.gamma)

    class PowerLawTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, gamma):
            mtransforms.Transform.__init__(self)
            self.gamma = gamma

        def transform_non_affine(self, a):
            return np.sign(a) * np.power(np.abs(a), self.gamma)

        #             return np.power(a, self.gamma)

        def inverted(self):
            return PowerLawScale.InvertedPowerLawTransform(self.gamma)

    class InvertedPowerLawTransform(mtransforms.Transform):
        input_dims = output_dims = 1

        def __init__(self, gamma):
            mtransforms.Transform.__init__(self)
            self.gamma = gamma

        def transform_non_affine(self, a):
            return np.sign(a) * np.power(np.abs(a), 1 / self.gamma)

        def inverted(self):
            return PowerLawScale.PowerLawTransform(self.gamma)


mscale.register_scale(PowerLawScale)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0.1, 100, 1000)

    plt.plot(x, x ** 2)

    plt.gca().set_xscale("power_law", gamma=2)