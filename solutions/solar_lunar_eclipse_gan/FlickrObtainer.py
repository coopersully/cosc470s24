import os
import requests
from flickrapi import FlickrAPI
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

# API Access
FLICKR_PUBLIC = os.environ.get('FLICKR_PUBLIC')
FLICKR_SECRET = os.environ.get('FLICKR_SECRET')

# Flickr Group Settings
group_id = '2986814@N24'

# Local Settings
dataset_dir = 'solar_lunar_eclipse_gan/flickr-dataset'

# Set up the Flickr API
flickr = FlickrAPI(FLICKR_PUBLIC, FLICKR_SECRET, format='parsed-json')

# Create a directory for your images
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)


def download_and_resize_images():
    photos = flickr.groups.pools.getPhotos(group_id=group_id, per_page=500)
    for photo in photos['photos']['photo']:
        url = f"https://farm{photo['farm']}.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}_c.jpg"
        response = requests.get(url)
        if response.status_code == 200:
            # Load image into Pillow
            image = Image.open(BytesIO(response.content))
            # Resize the image
            image = image.resize((32, 32))
            # Save the resized image
            image.save(f"{dataset_dir}/{photo['id']}.jpg", 'JPEG')
        print(f"Downloaded and resized {photo['id']}.jpg")


if __name__ == '__main__':
    download_and_resize_images()
