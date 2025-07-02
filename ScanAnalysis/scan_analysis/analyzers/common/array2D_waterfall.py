"""
Array2DWaterfall

Child to Array2DScanAnalyzer (./common/array2D_scan_analysis.py)
"""
# %% imports
from .array2D_scan_analysis import Array2DScanAnalyzer

# %% classes
class Array2DWaterfall(Array2DScanAnalyzer):

    def _postprocess_scan_parallel(self) -> None:
        # run super method
        super()._postprocess_scan_parallel()

        # create waterfall plot
        self.create_waterfall()

    def _create_waterfall(self, ):
        self.create_waterfall()

    def create_waterfall(self):

        plt.figure()

        #

