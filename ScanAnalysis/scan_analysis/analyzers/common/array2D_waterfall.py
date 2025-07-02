"""
Array2DWaterfall

Child to Array2DScanAnalyzer (./common/array2D_scan_analysis.py)
"""
# %% imports
import matplotlib
from networkx.algorithms.structuralholes import constraint

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from .array2D_scan_analysis import Array2DScanAnalyzer

# %% classes
class Array2DWaterfall(Array2DScanAnalyzer):

    def _postprocess_scan_parallel(self) -> None:
        # run super method
        super()._postprocess_scan_parallel()

        # create waterfall plot
        self._create_waterfall()

    def _postprocess_scan_interactive(self) -> None:
        # run super method
        super()._postprocess_scan_interactive()

        # create waterfall plot
        self._create_waterfall()

    def _create_waterfall(self):
        save_path = Path(self.path_dict["save"]) / f'{self.device_name}_waterfall.png'
        self.create_waterfall(save_path=save_path)
        self.display_contents.append(str(save_path))

    def create_waterfall(
            self,
            save_path: Optional[Path] = None,
            dpi: int = 150
    ):

        # compile image data
        image = [value['result']['analyzer_return_lineouts'][0] for _, value in self.binned_data.items()]
        scan_values = [round(value['value'], 2) for _, value in self.binned_data.items()]

        # initialize figure
        fig, ax = plt.subplots(dpi=dpi, constrained_layout=True)

        # display image
        im = ax.imshow(image, aspect='auto', cmap='viridis')

        cbar = plt.colorbar(im, ax=ax)

        ax.set_yticks(range(len(scan_values)))
        ax.set_yticklabels(scan_values)
        ax.set_ylabel(f"{self.scan_parameter}")

        plt.tight_layout()

        if save_path is None:
            filename = f'{self.device_name}_waterfall.png'
            save_path = Path(self.path_dict['save']) / filename

        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

        if self.flag_logging:
            logging.info(f"Saved waterfall plot as {save_path.name}.")




