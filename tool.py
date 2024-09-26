import glob
from PIL import Image, ImageDraw, ImageFont
import os

def merge_images_with_titles(image_files, output_path, rows, cols, titles):
    # Open all images
    images = [Image.open(image_file) for image_file in image_files]
    
    # Determine the size of each image
    widths, heights = zip(*(i.size for i in images))

    # Determine the size of the grid
    total_width = max(widths) * cols
    total_height = (max(heights) + 20) * rows  # Adding space for titles

    # Create a new image with a white background
    new_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    # Get a font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(new_image)

    # Paste each image into the grid
    for index, (image, title) in enumerate(zip(images, titles)):
        row = index // cols
        col = index % cols
        x_offset = col * max(widths)
        y_offset = row * (max(heights) + 20)  # Adjusting for title space
        new_image.paste(image, (x_offset, y_offset + 20))

        # Draw the title
        bbox = draw.textbbox((0, 0), title, font=font)
        title_width = bbox[2] - bbox[0]
        title_x = x_offset + (max(widths) - title_width) // 2
        title_y = y_offset
        if title == 'epoch 9':
            title = 'base'
        draw.text((title_x, title_y), title, fill="black", font=font)

    # Save the new image
    new_image.save(output_path)

if __name__ == "__main__":
    # Define the image files
    # image_files = [f"output/noise_0.05_lr_0.001/two_moons_4_subfunctions__results_4_{i}.png" for i in range(1, 10)] 
    # image_files = glob.glob("output/base/sample_number/two_moons_4_subfunctions__results_2_*.png")
    image_files = glob.glob("output/freeze/number_sampel/two_moons_4_subfunctions__results_2_*.png")
 
    # Define the titles for each image
    titles = [f"epoch {i}" for i in range(1, 11)]


    # Merge images into a 3x3 grid with titles
    merge_images_with_titles(image_files, "sample_number.png", rows=4, cols=4, titles=titles)
