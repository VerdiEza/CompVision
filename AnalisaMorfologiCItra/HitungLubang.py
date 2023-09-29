import numpy as np
from skimage import io, measure

# Baca citra biner dari URL atau upload dari komputer
img_url = "/content/Foto Verdi.JPG"  # Ganti dengan URL citra Anda atau path lokal
binary_image = io.imread(img_url, as_gray=True)

# Labeling wilayah-wilayah dalam citra biner
labeled_image = measure.label(binary_image, connectivity=2)

# Hitung jumlah lubang (wilayah dengan luas = 1)
hole_count = np.sum(labeled_image == 1)

# Tampilkan jumlah lubang
print(f"Jumlah Lubang: {hole_count}")
