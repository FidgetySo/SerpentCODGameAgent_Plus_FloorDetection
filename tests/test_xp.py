from PIL import Image
from mss import mss


def get_xp():
    img = Image.open("C:/Users/Fidgety/Pictures/bBo5dQgD2rHzYG7b7DVDgK.jpg")
    pixels = 0
    for pixel in img.getdata():
        if pixel == (173, 144, 41):
        	pixels += 1
    return pixels
pixel_count = get_xp()
print(pixel_count)