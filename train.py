from fastai.vision.all import *
import plotly.express as px
import sklearn.metrics as skm
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def get_labels(path, valid_idx):
    train_labels, valid_labels = [], []
    df = pd.read_csv(path/f'data-split-valid{valid_idx}.csv')
    for l in df[df.is_valid==False].labels:
        train_labels.extend(l.split(' '))
    for l in df[df.is_valid==True].labels:
        valid_labels.extend(l.split(' '))
    return set(train_labels), set(valid_labels)


def train_cnn(path, valid_idx, daug=False):
    if daug:
        df = pd.read_csv(path/f'data-split-valid{valid_idx}-daug.csv')
    else:
        df = pd.read_csv(path/f'data-split-valid{valid_idx}.csv')

    train_labels, valid_labels = get_labels(path, valid_idx)
    not_in_train = valid_labels.difference(train_labels)

    def get_x(r): return path/f'images-split-valid{valid_idx}'/r['fname']
    def get_y(r): return list(set(r['labels'].split(' ')).difference(not_in_train))

    def splitter(df):
        train = df.index[~df['is_valid']].tolist()
        valid = df.index[df['is_valid']].tolist()
        return train,valid

    dblock = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                    splitter=splitter,
                    get_x=get_x, 
                    get_y=get_y,
                    item_tfms=[Resize(350, method='pad', pad_mode='zeros')])
    dls = dblock.dataloaders(df, bs=32)

    learn = cnn_learner(dls, resnet50, metrics=partial(accuracy_multi, thresh=0.3))
    learn.fine_tune(20, base_lr=3e-3, freeze_epochs=4)

    if daug:
        learn.save(f'valid{valid_idx}-cnn-4fepochs-20epochs-lr0_003-daug')
    else:
        learn.save(f'valid{valid_idx}-cnn-4fepochs-20epochs-lr0_003')

    inp,preds,targs,decoded,losses = learn.get_preds(with_input=True, with_decoded=True, with_loss=True)
    print(skm.classification_report(targs, decoded, target_names=dls.vocab))


def train_mlknn(path, valid_idx, daug=False):
    labels = ['oven', 'refrigerator', 'dishwaser', 'kitchen_outlets', 'lighting', 'washer_dryer', 'microwave',
          'bathroom_gfi', 'electric_heat', 'stove', 'disposal', 'outlets_unknown', 'electronics', 'furance',
          'smoke_alarms', 'air_conditioning', 'miscellaeneous', 'subpanel']

    parameters = {'k': range(1,3), 's': [0.5, 0.7, 1.0]}
    if daug:
        X_train, y_train, X_test, y_test = np.load(path/f'X_train-split-valid{valid_idx}-daug.npy'), np.load(path/f'y_train-split-valid{valid_idx}-daug.npy'), np.load(path/f'X_test-split-valid{valid_idx}.npy'), np.load(path/f'y_test-split-valid{valid_idx}.npy')
    else:
        X_train, y_train, X_test, y_test = np.load(path/f'X_train-split-valid{valid_idx}.npy'), np.load(path/f'y_train-split-valid{valid_idx}.npy'), np.load(path/f'X_test-split-valid{valid_idx}.npy'), np.load(path/f'y_test-split-valid{valid_idx}.npy')

    clf = GridSearchCV(MLkNN(), parameters, scoring='f1_macro')
    clf.fit(X_train, y_train)
    print (clf.best_params_, clf.best_score_)

    classifier = MLkNN(k=clf.best_params_['k'], s=clf.best_params_['s'])

    # train
    classifier.fit(X_train, y_train)

    # predict
    predictions = classifier.predict(X_test)
    print(skm.classification_report(y_test, predictions.toarray(), target_names=labels))


def main(path):
    for i in range(1, 7):
        train_cnn(path, i)
        train_cnn(path, i, daug=True)
        train_mlknn(path, i)
        train_mlknn(path, i, daug=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path of the generated data')
    args = parser.parse_args()
    path = Path(args.path)
    main(path)
