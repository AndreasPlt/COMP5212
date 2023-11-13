import os

MIN_IMAGES = 100
DATA_DIR = os.path.join("..", "data", "kaggle_dataset")

def filter_countries(min_images=MIN_IMAGES, data_dir=DATA_DIR):
    valid_countries = []
    for subdir in os.listdir(data_dir):
        if os.path.isfile(os.path.join(data_dir, subdir)):
            continue
        
        num_images = len([name for name in os.listdir(os.path.join(data_dir, subdir)) 
                          if os.path.isfile(os.path.join(data_dir, subdir, name))
                          and os.path.splitext(name)[1] == ".jpg"])
        if num_images >= MIN_IMAGES:
            valid_countries.append(subdir)
            
    return valid_countries

def write_valid_countries(valid_countries, write_file_name):
    with open(write_file_name, "w") as write_file:
        for country in valid_countries:
            write_file.write(country + "\n")

def read_valid_countries(read_file_name):
    countries = []
    with open(read_file_name, "r") as read_file:
        for line in read_file:
            countries.append(line.strip())
    return countries

def main():
    valid_countries = filter_countries(MIN_IMAGES, DATA_DIR)
    write_valid_countries(valid_countries, os.path.join(DATA_DIR, "valid_countries.txt"))


if __name__ == "__main__":
    main()