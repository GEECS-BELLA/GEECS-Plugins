from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, get_type_hints, Callable, Any, Type, NamedTuple, Union
from pathlib import Path
from os import getenv

from docstring_parser import parse_from_object as parse_docstring_from_object
from docstring_parser import DocstringStyle

from importlib.metadata import version

# Q_ is used when converting pg_property from string to Quantity, and it needs
# to be the same object as used in the `ImageAnalyzer`s
from image_analysis import ureg, Q_, Quantity

from .frame import MainFrame

from image_analysis.utils import ROI

from wx import App
import wx.propgrid as pg

if TYPE_CHECKING:
    import wx
    import wx.grid

from ..scan_watch import ScanWatch
from ..scan_analyzer import ScanAnalyzer

class LivePostProcessingGUI(MainFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)

        # keep a single ScanAnalyzer in this GUI
        self.scan_analyzer = ScanAnalyzer()

        # load config from AppData
        if self.image_analyzers_config_cache_path.exists():
            self.load_image_analyzers_config(self.image_analyzers_config_cache_path)
        else:
            self.image_analyzers_config_cache_path.parent.mkdir()
            self.save_image_analyzers_config(self.image_analyzers_config_cache_path)

        self.populate_analyze_device_checklist()
        # set runID field to today's run folder
        self.m_runID_text.Value = datetime.now().strftime("%y_%m%d")

        self._print_welcome()

    def _print_welcome(self):
        print("*********************************\n"
              "* BELLA HTU Live Postprocessing *\n"
              "*********************************"
             )
        print(f"version {version('livepostprocessing')}")

    def populate_analyze_device_checklist(self):
        device_names: list[str] = list(self.scan_analyzer.image_analyzers.keys())
        self.m_analyze_device_checklist.SetItems(device_names)
        self.m_analyze_device_checklist.SetCheckedStrings([device_name for device_name in device_names if self.scan_analyzer.enable_image_analyzer[device_name]])

    @property
    def image_analyzers_config_cache_path(self) -> Path:
        """ The path where the current image analyzer config is saved. 

        Config is loaded from this file at startup, and 
        """
        return Path(getenv('APPDATA')) / 'LivePostProcessing' / "image_analyzer_config.ini"

    def save_image_analyzers_config(self, filename: Union[Path, str]):
        self.scan_analyzer.save_image_analyzer_config(filename)

    def load_image_analyzers_config(self, filename: Union[Path, str]):
        self.scan_analyzer.load_image_analyzer_config(filename)
        # check device names whose image analyzers are enabled
        print(self.scan_analyzer.enable_image_analyzer)
        self.populate_analyze_device_checklist()
        # if any device is selected, update the property grid
        if self.m_analyze_device_checklist.GetSelections():
            self._populate_property_grid(self.m_analyze_device_checklist.GetString(self.m_analyze_device_checklist.GetSelections()[0]))

    # ## event handlers

    # first set up converters between image analyzer properties and PGProperty
    # elements of the PropertyGrid
    class ImageAnalyzerParameterPGPropertyConverter(NamedTuple):
        # the PropertyGrid PGProperty subclass, such as pg.IntProperty that should
        # be used to 
        pg_property_subclass: Type[pg.PGProperty]

        # function that takes a image analyzer parameter value and returns a 
        # value corresponding to the pg_property_subclass
        parameter_value_to_pg_property_value: Callable[[Any], Any]

        # function that takes the value from the PGProperty and converts it 
        # back into the full-featured Python type
        pg_property_value_to_parameter_value: Callable[[Any], Any]

    image_analyzer_parameter_pg_property_map = {
        int: ImageAnalyzerParameterPGPropertyConverter(pg.IntProperty, int, int),
        float: ImageAnalyzerParameterPGPropertyConverter(pg.FloatProperty, float, float),
        str: ImageAnalyzerParameterPGPropertyConverter(pg.StringProperty, str, str),
        Path: ImageAnalyzerParameterPGPropertyConverter(pg.StringProperty, 
            lambda path: str(path) if path else '', 
            # property value is a string
            lambda pv: Path(pv) if pv else Path('NUL'),
        ),
        ROI: ImageAnalyzerParameterPGPropertyConverter(pg.ArrayStringProperty,
            lambda roi: [str(roi.top), str(roi.bottom), str(roi.left), str(roi.right)],
            # property value is a list of string
            lambda pvs: ROI(*[(int(pv) if pv.lower() != 'none' else None) for pv in pvs])
        ),
        Quantity: ImageAnalyzerParameterPGPropertyConverter(pg.StringProperty, str, Q_),
    }

    def _populate_property_grid(self, device_name: str):
        # get list of parameters and their types from the image analyzer's __init__
        image_analyzer = self.scan_analyzer.image_analyzers[device_name]
        image_analyzer_parameter_types = get_type_hints(image_analyzer.__init__)
        # get docstring parameter descriptions to add help text to properties.
        docstring_parameters = parse_docstring_from_object(image_analyzer.__init__, DocstringStyle.NUMPYDOC).params
        image_analyzer_parameter_descriptions = {
            parameter.arg_name: parameter.description 
            for parameter in docstring_parameters
        }

        self.m_image_analyzer_propertyGrid.Clear()
        image_analyzer_propertyGrid_page = self.m_image_analyzer_propertyGrid.AddPage("image_analyzer_properties")
        for parameter_name, parameter_type in image_analyzer_parameter_types.items():
            if parameter_type in self.image_analyzer_parameter_pg_property_map:
                # get the ImageAnalyzerParameterPGPropertyConverter instance for this type
                image_analyzer_parameter_pg_property_converter = self.image_analyzer_parameter_pg_property_map[parameter_type]
                # get parameter value from image analyzer and convert it for the PGProperty
                property_grid_value = image_analyzer_parameter_pg_property_converter.parameter_value_to_pg_property_value(getattr(image_analyzer, parameter_name))
                # add new PGProperty to the PropertyGrid
                property_grid_item = image_analyzer_propertyGrid_page.Append( 
                    image_analyzer_parameter_pg_property_converter.pg_property_subclass(
                        label=parameter_name, name=parameter_name, 
                        value=property_grid_value,
                    )
                )

                # get help string from docstring
                self.m_image_analyzer_propertyGrid.SetPropertyHelpString( 
                    property_grid_item, image_analyzer_parameter_descriptions.get(parameter_name, "")
                )

            else:
                print(f"Don't know how to make grid property from parameter {parameter_name} of type {parameter_type}")

        
    def m_analyze_device_checklist_OnCheckListBoxSelect( self, event: wx.CommandEvent ):
        """ Load image analyzer config.
        """
        self._populate_property_grid(event.GetString())


    def m_image_analyzer_propertyGrid_OnPropertyGridChanged( self, event: pg.PropertyGridEvent ):
        """ Update image analyzer property """

        device_name = self.m_analyze_device_checklist.GetString(self.m_analyze_device_checklist.GetSelection())
        image_analyzer = self.scan_analyzer.image_analyzers[device_name]

        property_name = event.GetPropertyName()
        image_analyzer_parameter_type = get_type_hints(image_analyzer.__init__)[property_name]
        # get the ImageAnalyzerParameterPGPropertyConverter instance for this type
        image_analyzer_parameter_pg_property_converter = self.image_analyzer_parameter_pg_property_map[image_analyzer_parameter_type]

        # convert new PGProperty value to full Python type
        image_analyzer_parameter_value = image_analyzer_parameter_pg_property_converter.pg_property_value_to_parameter_value(event.GetPropertyValue())

        setattr(image_analyzer, property_name, image_analyzer_parameter_value)

        self.save_image_analyzers_config(self.image_analyzers_config_cache_path)

    def m_analyze_device_checklist_OnCheckListBoxToggled( self, event: wx.CommandEvent ):
        """ Enable or disable image analyzer
        """
        self.scan_analyzer.enable_image_analyzer[event.GetString()] = self.m_analyze_device_checklist.IsChecked(event.GetInt())
        self.save_image_analyzers_config(self.image_analyzers_config_cache_path)

    def m_background_filePicker_OnFileChanged( self, event: wx.FileDirPickerEvent ):
        """ Update image analyzer background 
        """
        self.print_event(event)

    def m_run_live_analysis_Button_OnButtonClick( self, event: wx.CommandEvent ):
        """ Start live analysis
        """
        self.live_run_analyzer = ScanWatch(self.m_runID_text.Value, 
                                           scan_analyzer=self.scan_analyzer, 
                                          )
        self.live_run_analyzer.run(watch_folder_not_exist='raise')

        self.m_run_live_analysis_Button.Disable()
        self.m_stop_live_analysis_Button.Enable()

        self.m_analyze_device_checklist.Disable()
        self.m_image_analyzer_propertyGrid.Disable()

    def m_stop_live_analysis_Button_OnButtonClick( self, event: wx.CommandEvent ):
        self.live_run_analyzer.stop()
        self.live_run_analyzer = None

        self.m_run_live_analysis_Button.Enable()
        self.m_stop_live_analysis_Button.Disable()
        
        self.m_analyze_device_checklist.Enable()
        self.m_image_analyzer_propertyGrid.Enable()

    def m_run_scan_analysis_Button_OnButtonClick( self, event: wx.CommandEvent ):
        self.m_run_live_analysis_Button.Disable()
        self.m_run_scan_analysis_Button.Disable()
        self.SetStatusText("Running Scan analysis...")

        self.scan_analyzer.analyze_scan(self.m_runID_text.Value, int(self.m_scanNumber_textCtrl.Value))
        self.scan_analyzer.save_scan_metrics()

        self.m_run_live_analysis_Button.Enable()
        self.m_run_scan_analysis_Button.Enable()
        self.SetStatusText("Finished scan analysis")


    def m_loadconfig_button_OnButtonClick( self, event: wx.CommandEvent ):
        self.load_image_analyzers_config(self.m_config_filePicker.GetPath())
        self.save_image_analyzers_config(self.image_analyzers_config_cache_path)
        self.SetStatusText("Loaded configuration.")

    def m_saveconfig_button_OnButtonClick( self, event: wx.CommandEvent ):
        self.save_image_analyzers_config(self.m_config_filePicker.GetPath())
        self.SetStatusText("Saved configuration.")

    def print_event( self, event ):
        print(f"{type(event)=}\n{event.EventObject=}\n{event.EventType=}")

if __name__ == "__main__":
    app = App()
    frame = LivePostProcessingGUI()
    frame.Show()
    app.MainLoop()
