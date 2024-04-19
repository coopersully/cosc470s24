import os
import imageio


def create_gif_from_images(output_gif_name="output.gif", frame_duration=0.5):
    """
    Creates a GIF from images in the current directory starting with 'image_at_epoch'.

    Parameters:
        output_gif_name (str): Name of the output GIF file.
        frame_duration (float): Duration of each frame in the GIF in seconds.
    """
    # Get all files in the current directory
    files = os.listdir('.')

    # Filter files that start with 'image_at_epoch' and end with an image extension
    image_files = sorted(
        [f for f in files if f.startswith('image_at_epoch') and f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    # Ensure there are images to process
    if not image_files:
        print("No images found starting with 'image_at_epoch'. Exiting.")
        return

    # Create a list to hold the images
    images = []

    # Load each image file
    for filename in image_files:
        images.append(imageio.imread(filename))

    # Write out the GIF
    imageio.mimsave(output_gif_name, images, duration=frame_duration)

    print(f"GIF created successfully: {output_gif_name}")


# Call the function to create the GIF
create_gif_from_images("eclipse_generations.gif", 0.5)
