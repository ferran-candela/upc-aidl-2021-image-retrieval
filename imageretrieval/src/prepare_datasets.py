import os
import torch
import csv

import pandas as pd

from config import FoldersConfig

def txt_to_csv(input_path, output_path):
    with open(input_path, 'r') as in_file:
        stripped = (line.strip() for line in in_file)
        lines = (line.split() for line in stripped if line)
        with open(output_path, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(lines)


def get_categories_and_path(input_path, output_path):
    with open(input_path, 'r') as in_file:
        reader = csv.reader(in_file)
        next(reader)
        row0 = next(reader)
        with open(output_path, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(["path", "deep_fashion_category_name", "dataset"])
            for r in reader:
                split_r = r[0].split('_')[-2]
                category = split_r.split('/')[-2]
                r.append(r[0])
                r.append(category)
                r.append('deep_fashion')
                writer.writerow( (r[2], r[3], r[4]) )


def add_column_with_article_type_equivalence(deep_fashion, map_to_product_fashion, output_path):
    deep_fashion_df = pd.read_csv(deep_fashion, error_bad_lines=False)
    map_to_product_fashion_df = pd.read_csv(map_to_product_fashion)

    deep_fashion_with_article_type = deep_fashion_df.merge(map_to_product_fashion_df, on='deep_fashion_category_name', how='left')

    deep_fashion_with_article_type.to_csv(output_path)



def prepare_datasets():
    resources = FoldersConfig.RESOURCES_DIR
    list_categories_path = resources + 'deep_fashion/list_category_img.txt'
    list_categories_output_path = resources + 'deep_fashion/list_category_img.csv'
    path_category_dataset = resources + 'deep_fashion/path_category_dataset.csv'
    map_to_product_fashion_path = resources + 'map_deep_fashion_to_product_fashion.csv'
    deep_fashion_with_article_type_path = resources + 'deep_fashion/deep_fashion_with_article_type.csv'
    
    
    if not os.path.exists(list_categories_output_path):
        txt_to_csv(list_categories_path, list_categories_output_path)

    if not os.path.exists(path_category_dataset):
        get_categories_and_path(list_categories_output_path, path_category_dataset)

    if not os.path.exists(deep_fashion_with_article_type_path):
        add_column_with_article_type_equivalence(path_category_dataset, map_to_product_fashion_path, deep_fashion_with_article_type_path)
    


if __name__ == "__main__":
    prepare_datasets()