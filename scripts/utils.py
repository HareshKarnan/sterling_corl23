import numpy as np
import albumentations as A

def process_feet_data(feet):
    filtered_feet = []
    for fi in range(feet.shape[0]):
        contacts = [feet[fi, 3], feet[fi, 6+3], feet[fi, 12+3], feet[fi, 18+3]]
        mu, std = [feet[fi, 4], feet[fi, 10], feet[fi, 16], feet[fi, 22]], [feet[fi, 5], feet[fi, 11], feet[fi, 17], feet[fi, 23]]
        
        for i in range(4):
            if contacts[i] != 1:
                mu[i], std[i] = -1, -1
                    
        # np remove all mu and std values from feet[fi, :]
        curr = np.delete(feet[fi, :], [3, 4, 5, 9, 10, 11, 15, 16, 17, 21, 22, 23])
        curr = np.hstack((curr, mu, std))
        
        filtered_feet.append(curr)
    return  np.asarray(filtered_feet)

def get_transforms():
    return A.Compose([
                A.Flip(always_apply=False, p=0.5),
                # A.CoarseDropout(always_apply=False, p=1.0, max_holes=5, max_height=16, max_width=16, min_holes=1, min_height=2, min_width=2, fill_value=(0, 0, 0), mask_fill_value=None),
                # A.AdvancedBlur(always_apply=False, p=0.1, blur_limit=(3, 7), sigmaX_limit=(0.2, 1.0), sigmaY_limit=(0.2, 1.0), rotate_limit=(-90, 90), beta_limit=(0.5, 8.0), noise_limit=(0.9, 1.1)),
                A.ShiftScaleRotate(always_apply=False, p=0.75, shift_limit_x=(-0.1, 0.1), shift_limit_y=(-0.1, 0.1), scale_limit=(-0.1, 2.0), rotate_limit=(-21, 21), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box'),
                A.Perspective(always_apply=False, p=0.5, scale=(0.025, 0.25), keep_size=1, pad_mode=0, pad_val=(0, 0, 0), mask_pad_val=0, fit_output=0, interpolation=3),
                # A.ISONoise(always_apply=False, p=0.5, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
                # A.RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
                # A.ToGray(always_apply=False, p=0.5),
            ])
    