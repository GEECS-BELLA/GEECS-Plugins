import numpy as np
from typing import Any, Optional


def add_beam_analysis(beam_analysis: dict[str, Any], analysis_dict: dict[str, Any],
                      pos_short_names: list[str], pos_long_names: list[str],
                      index: int, init_size: Optional[int] = None):
    summary: dict[str, Any] = analysis_dict['analyses_summary']
    targets: dict[str, Any] = summary['targets']

    if not pos_short_names:
        for im_analysis in analysis_dict['image_analyses']:
            if 'positions' in im_analysis:
                pos_short_names = [pos[-1] for pos in im_analysis['positions']]
                pos_long_names = [pos for pos in im_analysis['positions_labels']]
                break

    for pos in pos_short_names:
        if f'{pos}_deltas_avg_imgs' not in beam_analysis and init_size:
            tmp = np.zeros((init_size, 2))
            tmp[:] = np.nan
            beam_analysis[f'{pos}_deltas_avg_imgs'] = tmp.copy()  # Dx, Dy [mm]
            beam_analysis[f'{pos}_deltas_means'] = tmp.copy()
            beam_analysis[f'{pos}_deltas_stds'] = tmp.copy()
            beam_analysis[f'{pos}_fwhm_means'] = tmp.copy()
            beam_analysis[f'{pos}_fwhm_stds'] = tmp.copy()
            beam_analysis[f'pos_pix_{pos}']: list[np.ndarray] = []
            beam_analysis[f'{pos}_mean_pos_pix'] = tmp.copy()
            beam_analysis[f'{pos}_std_pos_pix'] = tmp.copy()
            beam_analysis['target_xy'] = tmp.copy()
            beam_analysis['target_um_pix'] = np.ones((init_size,), dtype=float)

        if targets and (f'avg_img_{pos}_delta' in targets):
            beam_analysis[f'{pos}_deltas_avg_imgs'][index, :] = targets[f'avg_img_{pos}_delta']
            beam_analysis[f'{pos}_deltas_means'][index, :] = targets[f'target_deltas_{pos}_mean']
            beam_analysis[f'{pos}_deltas_stds'][index, :] = targets[f'target_deltas_{pos}_std']

            beam_analysis['target_xy'][index, :] = targets['target_xy']
            beam_analysis['target_um_pix'][index] = targets['target_um_pix']

        if summary and (f'mean_pos_{pos}_fwhm_x' in summary):
            beam_analysis[f'pos_pix_{pos}'].append(summary[f'scan_pos_{pos}'])
            beam_analysis[f'{pos}_mean_pos_pix'][index, :] = summary[f'mean_pos_{pos}']
            beam_analysis[f'{pos}_std_pos_pix'][index, :] = summary[f'std_pos_{pos}']

            beam_analysis[f'{pos}_fwhm_means'][index, 1] = \
                summary[f'mean_pos_{pos}_fwhm_x'] * beam_analysis['target_um_pix'][index]
            beam_analysis[f'{pos}_fwhm_means'][index, 0] = \
                summary[f'mean_pos_{pos}_fwhm_y'] * beam_analysis['target_um_pix'][index]
            beam_analysis[f'{pos}_fwhm_stds'][index, 1] = \
                summary[f'std_pos_{pos}_fwhm_x'] * beam_analysis['target_um_pix'][index]
            beam_analysis[f'{pos}_fwhm_stds'][index, 0] = \
                summary[f'std_pos_{pos}_fwhm_y'] * beam_analysis['target_um_pix'][index]

    return beam_analysis, pos_short_names, pos_long_names
