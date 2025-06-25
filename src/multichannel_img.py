import numpy as np

RGB_BANDS = (5, 3, 1)
NIR_R_G = (7, 5, 3)


def broad_band(all_bands: np.ndarray, no_data: np.ndarray) -> np.ndarray:
    # Natural colour broad band, log scaled
    #     red_recipe = 0.16666 * all_bands[5] + 0.66666 * all_bands[5] \
    #                  + 0.08333 * all_bands[5] + 0.4 * all_bands[6] + 0.4 * all_bands[7]
    #     green_recipe = 0.16666 *  all_bands[2] + 0.66666 *  all_bands[3] \
    #                    + 0.16666 *  all_bands[4]
    #     blue_recipe = 0.16666 *  all_bands[0] + 0.66666 *  all_bands[0] \
    #                    + 0.16666 *  all_bands[1]

    red_recipe = np.mean(all_bands[5:], axis=0)
    green_recipe = np.mean(all_bands[2:5], axis=0)
    blue_recipe = np.mean(all_bands[:2], axis=0)

    rgb_log = np.dstack((np.log10(1.0 + red_recipe), np.log10(1.0 + green_recipe), np.log10(1.0 + blue_recipe)))

    mins = np.array([rgb_log[:, :, i][~no_data].min() for i in range(3)])

    rgb_log -= mins
    rgb_log /= rgb_log.max(axis=(0, 1))

    rgb_log[no_data] = 0.0

    return rgb_log


# Tristimulus
def tristimulus(all_bands: np.ndarray, no_data: np.ndarray) -> np.ndarray:
    #     red_recipe = np.log10(1.0 + 0.01 * band_dict['Oa01_radiance'] + 0.09 * band_dict['Oa02_radiance']
    #                           + 0.35 * band_dict['Oa03_radiance'] + 0.04 * band_dict['Oa04_radiance']
    #                           + 0.01 * band_dict['Oa05_radiance'] + 0.59 * band_dict['Oa06_radiance']
    #                           + 0.85 * band_dict['Oa07_radiance'] + 0.12 * band_dict['Oa08_radiance']
    #                           + 0.07 * band_dict['Oa09_radiance'] + 0.04 * band_dict['Oa10_radiance'])

    red = np.log10(
        1.0
        # all_bands[0] * (0.09 + 0.35) +
        # all_bands[1] * 0.04 +
        # all_bands[2] * 0.01 +
        # all_bands[3] * 0.59 +
        + all_bands[4] * 0.85
        + all_bands[5] * (0.12 + 0.9 + 0.04)
        + all_bands[6] * 1.0
        + all_bands[7] * 1.0
    )
    #     green_recipe = np.log10(1.0 + 0.26 * band_dict['Oa03_radiance'] + 0.21 * band_dict['Oa04_radiance']
    #                             + 0.50 * band_dict['Oa05_radiance'] + band_dict['Oa06_radiance']
    #                             + 0.38 * band_dict['Oa07_radiance'] + 0.04 * band_dict['Oa08_radiance']
    #                             + 0.03 * band_dict['Oa09_radiance'] + 0.02 * band_dict['Oa10_radiance'])

    green = np.log10(
        1.0
        + all_bands[0] * 0.26
        + all_bands[1] * 0.21
        + all_bands[2] * 0.50
        + all_bands[3] * 0.38
        + all_bands[4] * 0.04
        + all_bands[5] * (0.03 + 0.02)
    )
    #     blue_recipe = np.log10(1.0 + 0.07 * band_dict['Oa01_radiance'] + 0.28 * band_dict['Oa02_radiance']
    #                            + 1.77 * band_dict['Oa03_radiance'] + 0.47 * band_dict['Oa04_radiance']
    #                            + 0.16 * band_dict['Oa05_radiance'])

    blue = np.log10(1.0 + all_bands[0] * (0.28 + 1.77) + all_bands[1] * 0.27 + all_bands[2] * 0.16)

    rgb_tri = np.dstack((red, green, blue))
    mins = np.array([rgb_tri[:, :, i][~no_data].min() for i in range(3)])

    rgb_tri[no_data] = mins

    rgb_tri -= mins
    rgb_tri /= rgb_tri.max(axis=(0, 1))

    rgb_tri[no_data] = 0.0

    return rgb_tri


def rgb_log_image(all_bands: np.ndarray, no_data: np.ndarray) -> np.ndarray:
    img = np.array([np.log10(1.0 + all_bands[idx]) for idx in RGB_BANDS]).transpose((1, 2, 0)).copy()
    img -= img[~no_data].min(axis=(0, 1))
    img /= img[~no_data].max(axis=(0, 1))
    img[no_data] = 0.0

    return img


def false_color_log(all_bands: np.ndarray, no_data: np.ndarray) -> np.ndarray:
    img = np.zeros((*no_data.shape, 3), dtype=np.float32)
    if len(all_bands) == 8:
        img[:, :, 0] = all_bands[NIR_R_G[0]]
        img[:, :, 1] = all_bands[NIR_R_G[1]]
        img[:, :, 2] = all_bands[NIR_R_G[2]]
    else:
        img[:, :, 0] = all_bands[3]
        img[:, :, 1] = all_bands[2]
        img[:, :, 2] = all_bands[1]

    img = np.log10(1 + img)
    img -= img[~no_data].min()
    img /= img[~no_data].max()
    img[no_data] = 0.0

    return img
