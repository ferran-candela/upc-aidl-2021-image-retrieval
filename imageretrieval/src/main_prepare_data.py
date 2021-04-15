import os
import pandas as pd

config = {
        "dataset_base_dir" : "/Users/melaniasanchezblanco/Documents/UPC_AIDL/Project/Fashion_Product_Full/",
        "dataset_labels_file" : "/Users/melaniasanchezblanco/Documents/UPC_AIDL/Project/Fashion_Product_Full/styles.csv",
        "debug" : "False" # True/False
    }


if __name__ == "__main__":

    labels_file = config["dataset_labels_file"]

    # Work directory
    dataset_base_dir = config["dataset_base_dir"]

    df_dataset = pd.read_csv(labels_file, error_bad_lines=False)

    print(df_dataset.count())

    print(df_dataset.masterCategory.unique())

    different_clothes = ['Bra', 'Kurtas', 'Briefs', 'Sarees', 'Innerwear Vests', 
                        'Kurta Sets', 'Shrug', 'Camisoles', 'Boxers', 'Dupatta', 
                        'Capris', 'Bath Robe', 'Tunics', 'Trunk', 'Baby Dolls', 
                        'Kurtis', 'Suspenders', 'Robe', 'Salwar and Dupatta', 
                        'Patiala', 'Stockings', 'Tights', 'Churidar', 'Shapewear',
                        'Nehru Jackets', 'Salwar', 'Rompers', 'Lehenga Choli',
                        'Clothing Set', 'Belts']

    is_clothes = df_dataset['masterCategory'] == 'Apparel'
    is_shoes = df_dataset['masterCategory'] == 'Footwear'
    is_differenet_clothes = df_dataset['articleType'].isin(different_clothes)

    df_clothes_shoes = df_dataset[(is_clothes | is_shoes) & ~is_differenet_clothes]

    print(df_clothes_shoes.count())

    print(df_clothes_shoes.articleType.unique().size)

    df_clothes_shoes.to_csv(os.path.join(dataset_base_dir, "clothes_shoes.csv"),index=False)