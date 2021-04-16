import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt  

from imageretrieval.src.models import ModelManager
from imageretrieval.src.features import FeaturesManager
from imageretrieval.src.dataset import DatasetManager
from imageretrieval.src.config import DebugConfig, FoldersConfig, DeviceConfig, RetrievalEvalConfig, ModelTrainConfig

device = DeviceConfig.DEVICE
DEBUG = DebugConfig.DEBUG

def prepare_data(dataset_base_dir, labels_file, process_dir, train_size, validate_test_size, clean_process_dir):
    dataset_manager = DatasetManager()      

    train_df, test_df, validate_df = dataset_manager.split_dataset(dataset_base_dir=dataset_base_dir,
                                                    original_labels_file=labels_file,
                                                    process_dir=process_dir,
                                                    clean_process_dir=clean_process_dir,
                                                    train_size=train_size,
                                                    fixed_validate_test_size=validate_test_size
                                                    )
        
    train_df.reset_index(drop=True, inplace=True)
    if DEBUG:print(train_df.head(10))
    if not test_df is None:
        test_df.reset_index(drop=True, inplace=True)
        validate_df.reset_index(drop=True, inplace=True)

    return train_df, test_df, validate_df

def compute_tsne(features):
    print('\nComputing tsne...')
    n_iter = 800
    tsne = TSNE(n_components=2, random_state=0, n_iter=n_iter)
    return tsne.fit_transform(features)

def print_tsne(features, full_df, tsnePath, feature_type):
    feats_2d = compute_tsne(features)         

    labels_df = full_df[['articleType', 'articleTypeEncoded']]

    labels_unique = labels_df.drop_duplicates(subset=['articleType', 'articleTypeEncoded'], keep='last')

    encoded_labels = labels_unique['articleTypeEncoded'].values
    num_classes = len(encoded_labels)

    mapper = {}

    for i, label in enumerate(encoded_labels):
        mapper[label] = i

    labels_df = labels_df.replace(mapper)

    fig = plt.figure(figsize=(10,8))
    plt.scatter(feats_2d[:,0], feats_2d[:,1], c=labels_df['articleTypeEncoded'].values, cmap=plt.cm.get_cmap("jet", num_classes), s=5)
    plt.clim(-0.5, num_classes - 0.5)
    cbar = plt.colorbar(ticks=range(num_classes))
    cbar.ax.set_yticklabels(labels_unique['articleType'].values)

    if not os.path.exists(tsnePath):
        os.makedirs(tsnePath)

    features_size = features[0].shape[0]
    fig.savefig(os.path.join(tsnePath, 'tsne_' + feature_type + '_' + str(features_size) + '.png'), bbox_inches='tight')


def print_models_tsne():
    model_manager = ModelManager(device, FoldersConfig.WORK_DIR)
    features_manager = FeaturesManager(device, model_manager)

    model_names = model_manager.get_model_names()

    # The path of original dataset
    dataset_base_dir = FoldersConfig.DATASET_BASE_DIR
    labels_file = FoldersConfig.DATASET_LABELS_DIR

    # Work directory
    work_dir = FoldersConfig.WORK_DIR
    train_df, test_df, validate_df = prepare_data(dataset_base_dir=dataset_base_dir,
                                        labels_file=labels_file,
                                        process_dir=work_dir,
                                        clean_process_dir=False,
                                        train_size='all',
                                        validate_test_size=0)
    num_queries = RetrievalEvalConfig.MAP_N_QUERIES

    for model_name in model_names:
        try:
            print('\n\n## Printing t-SNE for model ', model_name, "with num_queries=", str(num_queries))

            model_path = model_manager.get_model_dir(model_name=model_name)
            tsnePath = os.path.join(model_path, 'tsne')


            if(features_manager.is_normalized_feature_saved(model_name)):
                print('\n\n## Using NormalizedFeatures ', model_name)

                usedfeatures = 'NormalizedFeatures'

                # LOAD FEATURES
                print('\nLoading features from checkpoint...')
                loaded_model_features = features_manager.load_from_norm_features_checkpoint(model_name)
                features = loaded_model_features['normalized_features']
                full_df = loaded_model_features['data']
                print_tsne(features, full_df, tsnePath, usedfeatures)


            if(features_manager.is_aqe_feature_saved(model_name)):
                print('\n\n## Using AQE features ', model_name)

                # LOAD AVERAGE QUERY EXPANSION FEATURES
                usedfeatures = 'AQEFeatures'

                print('\nLoading AQE features from checkpoint...')
                loaded_model_features = features_manager.load_from_aqe_features_checkpoint(model_name)
                features = loaded_model_features['aqe_features']
                full_df = loaded_model_features['data']
                print_tsne(features, full_df, tsnePath, usedfeatures)
                
        except Exception as e:
            print('\n', e)


if __name__ == "__main__":
    print_models_tsne()