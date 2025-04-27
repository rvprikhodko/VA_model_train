import numpy as np
import SimpleITK as sitk
import ants
from skimage.filters import frangi
from skimage.util import img_as_float, img_as_ubyte

def denoise_with_ants(image):
    ants_image = ants.from_numpy(np.array(image))
    imagenoise = ants_image + np.random.randn(*ants_image.shape).astype('float32')
    denoised_image = ants.denoise_image(imagenoise, ants.get_mask(ants_image))
    return denoised_image.numpy()

def denoise_with_sitk(image):
    sitk_image = sitk.GetImageFromArray(np.array(image))
    denoised_image = sitk.CurvatureFlow(
        image1=sitk_image,
        timeStep=0.25,
        numberOfIterations=25
    )
    return sitk.GetArrayFromImage(denoised_image)


def enhance_vessels_with_frangi(image_array):
    float_image = img_as_float(image_array)
    enhanced_image = frangi(
        float_image,
        sigmas=range(1, 5),
        black_ridges=True,
        mode='constant',
    )
    enhanced_image = (enhanced_image - enhanced_image.min()) / (enhanced_image.max() - enhanced_image.min() + 1e-8)
    return img_as_ubyte(enhanced_image)