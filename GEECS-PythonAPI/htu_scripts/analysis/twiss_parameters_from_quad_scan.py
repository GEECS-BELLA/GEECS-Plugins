from geecs_python_api.controls.api_defs import ScanTag
from geecs_python_api.tools.images.filtering import FiltersParameters
from geecs_python_api.analysis.scans.quad_scan_analysis import QuadAnalysis


# parameters
# --------------------------------------------------------------------------
scan_tag = ScanTag(2024, 5, 9, 56)

camera_of_interest = 'A3'
quad_used = 3
quad_2_screen = 2.126  # [m]

save_plots = True
save_data = True


# scan analysis
# --------------------------------------------------------------------------
quad_analysis = QuadAnalysis(scan_tag, quad_used, camera_of_interest,
                             fwhms_metric='median', quad_2_screen=quad_2_screen)

filters = FiltersParameters(contrast=1.333, hp_median=2, hp_threshold=3., denoise_cycles=0, gauss_filter=5.,
                            com_threshold=0.8, bkg_image=None, box=True, ellipse=False)

analysis_path = quad_analysis.analyze(None, initial_filtering=filters, ask_rerun=False, blind_loads=True,
                                      store_images=False, store_scalars=False, save_plots=save_plots, save=save_data)

save_plots_dir = quad_analysis.scan_data.get_analysis_folder() if save_plots else None
quad_analysis.render_twiss(physical_units=True, save_dir=save_plots_dir)

print('done')
