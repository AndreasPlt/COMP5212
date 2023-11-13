import os
import random
from util.filter_countries import read_valid_countries

def create_manifest(train_split, dev_split, test_split, output_dir, root_dir, subfolders, seed=187):
    """
    Creates the manifest files for the train, dev and test split.
    """
    assert train_split > 0
    assert dev_split > 0
    assert test_split > 0
    assert train_split + dev_split + test_split == 1


    train_out = os.path.join(output_dir, "train.tsv")
    dev_out = os.path.join(output_dir, "dev.tsv")
    test_out = os.path.join(output_dir, "test.tsv")

    train_tsv = open(train_out, "w")
    dev_tsv = open(dev_out, "w")
    test_tsv = open(test_out, "w")

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        subfolder_files = []

        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)
            if not os.path.isfile(image_path):
                continue
            if image_name.endswith(".jpg"):
                # append full path
                subfolder_files.append(os.path.join(subfolder_path, image_name))

        # split subfolder_files into train, dev and test
        num_files = len(subfolder_files)
        num_train = int(train_split * num_files)
        num_dev = int(dev_split * num_files)
        num_test = int(test_split * num_files)
        diff = num_files - (num_train + num_dev + num_test)
        if diff > 0:
            # distribute the remaining files to the splits
            num_train += (diff // 3) + (diff % 3)
            num_dev += diff // 3
            num_test += diff // 3
        elif diff < 0:
            assert False, f"This should not happen: {num_train} + {num_dev} + {num_test} > {num_files}"
        
        assert num_train + num_dev + num_test == num_files
        # shuffle the files
        random.seed(seed)
        random.shuffle(subfolder_files)
        
        # write to the manifest files
        for i in range(num_train):
            train_tsv.write(f"{subfolder_files[i]}\n")
        for i in range(num_train, num_train + num_dev):
            dev_tsv.write(f"{subfolder_files[i]}\n")
        for i in range(num_train + num_dev, num_train + num_dev + num_test):
            test_tsv.write(f"{subfolder_files[i]}\n")
        
    train_tsv.close()
    dev_tsv.close()
    test_tsv.close()

def main():
    root_dir = "../../data/kaggle_dataset"
    valid_countries_file = "../../data/kaggle_dataset/valid_countries.txt"
    subfolders = read_valid_countries(valid_countries_file)
    create_manifest(
        train_split=0.8,
        dev_split=0.1,
        test_split=0.1,
        output_dir="../../data/kaggle_dataset",
        root_dir=root_dir,
        subfolders=subfolders
    )

if __name__ == "__main__":
    main()