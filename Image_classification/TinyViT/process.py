import os
import shutil

def move_images(source_dir, dest_dir):
    # Ensure destination directory exists, create if not
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Iterate over files in source directory
    for filename in os.listdir(source_dir):
        # Check if the file is an image (you can extend the list of image extensions)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Construct absolute paths
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)
            
            # Move the image file to the destination directory
            shutil.move(source_file, dest_file)
            print(f"Moved {filename} to {dest_dir}")

# for file_path in os.listdir("./datasets/ImageNet/train"):
#     class_path = os.path.join(f"./datasets/ImageNet/train/{file_path}","images")
#     # os.rmdir(class_path)
#     source_directory = class_path
#     destination_directory = f"./datasets/ImageNet/train/{file_path}"
    # move_images(source_directory, destination_directory)


with open("./datasets/ImageNet/val/val_annotations.txt", "r") as file:
    list = []
    for line in file:
        index = line.strip().index("G")
        image_name, folder_name = line.split(" ")[0][:index+1],line.split(" ")[0][index+1:index+11]
        print(image_name,folder_name)
        folder_name = folder_name.replace('\t','')
        # if folder_name not in list:
        #     os.mkdir(f"./datasets/ImageNet/test/{folder_name}")
        #     shutil.move(f"./datasets/ImageNet/test/images/{image_name}", f"./datasets/ImageNet/val/{folder_name}")
        #     list.append(folder_name)
        # else:
        shutil.move(f"./datasets/ImageNet/test/images/{image_name}", f"./datasets/ImageNet/val/{folder_name}")

        
