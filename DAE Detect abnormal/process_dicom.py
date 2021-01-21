import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image
import matplotlib.pyplot as plt

class DicomLoader():
    def __init__(self,load_path):
        self.dicom = pydicom.read_file(load_path)
        self.data_friendly = None
        self.voi_lut = None
        self.fix_monochrome = None
    def _get_friendly_data(self):
        if self.data_friendly is not None:
            return self.data_friendly
        else:
            return self.read_xray()
    
    def read_xray(self, voi_lut = True, fix_monochrome = True):
        if self.data_friendly is not None and self.voi_lut == voi_lut and self.fix_monochrome == fix_monochrome:
            return self.data_friendly
        else:
            self.voi_lut = voi_lut
            self.fix_monochrome = fix_monochrome
        dicom = self.dicom
        # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
        if voi_lut:
            data = apply_voi_lut(dicom.pixel_array, dicom)
        else:
            data = dicom.pixel_array
        # depending on this value, X-ray may look inverted - fix that:
        if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
        self.data_friendly = data
        return data
    
    def resize(self,size, keep_ratio=False, resample=Image.LANCZOS):
        # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
        im = Image.fromarray(self._get_friendly_data())
        
        if keep_ratio:
            im.thumbnail((size, size), resample)
        else:
            im = im.resize((size, size), resample)
        return np.array(im.getdata())

    def imshow(self,figsize = (12,12)):
        plt.figure(figsize = figsize)
        plt.imshow(self._get_friendly_data(), 'gray')