"""
Modeler

Author: Wataoka Koki
E-mail: koki.wataoka@st.osaka-cu.ac.jp
"""

import os
from collections import OrderedDict

class Modeler():

    """
    modelerの使い方

    1. modelerオブジェクトを生成する。
    2. add関数でmodel達を追加する。(modelはコンパイル済み)
    3. start関数でmodel達を一つずつ学習, 評価, 保存を繰り返す。
    """

    def __init__(self):
        """
        modelsディクショナリについて
        key: model (modelオブジェクト)
        value: acc (float)
        """
        self.models = {}


    def add(self, model):
        self.models[model] = 0.0


    def start(self, data, batch_size=128, epochs=500, verbose=1):

        """
        モデル達を一つずつ学習させる。

        Args:
            data: (x_train, y_train), (x_test, y_test)という構造のタプル
            batch_size: バッチサイズ, int
            epochs: エポックの回数, int
            verbose: 途中出力の設定(1がおすすめ), int
        Return:
            無し
        """

        (x_train, y_train), (x_test, y_test) = data

        for model in self.models:
            model.summary()
            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=verbose)
            self.models[model] = model.evaluate(x_test, y_test)[0]




    def save(self, models_dir="./models/", n_save=1):

        """
        モデルオブジェクト達を保存する。上位から順に保存される。

        Args:
            models_dir: 保存するディレクトリ, string
            n_save: 保存するモデルの数, int
        Return:
            無し
        """

        models = OrderedDict(sorted(self.models.items(), key=lambda x:x[1], reverse=True))

        if not os.path.exists(models_dir):
            os.mkdir(models_dir)

        for n, model in enumerate(models):
            if n < n_save:
                model.save(models_dir + "model_" + str(n) + ".h5")
            else:
                break
