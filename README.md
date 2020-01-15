# object-detection-api-procedure

### object-detection-api　をfine-tunning

・独自用意した画像にて学習

・学習したモデルで推論


-----
### Object Detection APIのリポジトリをクローン

```
git clone https://github.com/tensorflow/models
```

-----
### Pathを通しておく

#### From tensorflow/models/research/

```
cd model/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim  # 
echo $PYTHONPATH

>>>:/home/user1/workspace/models/research:/home/user1/workspace/models/research/slim
```

-----

### Protobufのコンパイル
Tensorflow Object Detection APIは、Protobufsを使用してモデルとトレーニングパラメーターを構成します。

フレームワークを使用するには、Protobufライブラリをコンパイルする必要があります。これは、tensorflow/models/research/ディレクトリから次のコマンドを実行して実行する必要があります。

#### Download and install the 3.0 release of protoc, then unzip the file.

#### From tensorflow/models/research/

```
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip

## Run the compilation process again, but use the downloaded version of protoc

./bin/protoc object_detection/protos/*.proto --python_out=.

#ファイルが生成される
```


----

## 学習用の画像収集

サイズは特に指定無し 
`./images` ディレクトリ以下へ配置しておく

（参考）
```
./images/100.png
./images/200.png
```

----
## 学習用に画像へ矩形アノテーションを行い、XY座標の入ったデータを準備

`.csv`形式でファイル用意

カラム名は 'class','img_path', 'xmin', 'ymin', 'xmax', 'ymax'

（参考）
```
class,filename,xmin,ymin,xmax,ymax
room,100.png,2376,649,2405,667
room,100.png,2066,584,2125,609
room,100.png,1742,523,1837,598
room,200.png,1550,533,1621,560
room,200.png,1236,616,1293,643
```


### 学習用、テスト評価用に2種作成しておく

`./labels` ディレクトリ以下へ配置しておく

（参考）
```
./labels/train_labels.csv  #学習用アノテーションデータ
./labels/test_labels.csv  #テスト用アノテーションデータ
```


----
#### class名データの作成
Make a new file `object-detection.pbtxt` which looks like this:
```
item {
  id: 1
  name: 'room'
}
```

-----
### label.csvからtfrecord形式へ変換する

`generate_tfrecord.py`

### From tensorflow/models/reserch/
```
cd ./tensorflow/models/reserch/
```


### Create train data:

```
python generate_tfrecord.py --csv_input=./labels/train_labels.csv  --output_path=./path/to/train.record --image_dir ../images
```

### Create test data:
```
python generate_tfrecord.py --csv_input=./labels/test_labels.csv  --output_path=./path/to/test.record --image_dir ../images
```


-----
### fine-tuneする学習済みモデルを決め、configを書き直す

今回はfaster rcnn でいく

```
cd ./models/research/object_detection/faster_rcnn_nas_coco/
cp ./pipeline.config ./pipeline.config_backup
```

`pipeline.config`を編集する

```
#before
num_classes: 90
#after
num_classes: 1

#before
height: 1200
width: 1200
#after
height: 300
width: 300

#before
schedule {
  step: 0
  learning_rate: 0.000300000014249
}

#after
#ステップごとの学習率スケジュール　initial_learning_rate:が既に指定されているので、step:0箇所でも指定されると学習時にエラー出る。コメントアウトした
#schedule {
#  step: 0
#  learning_rate: 0.000300000014249
#}


#before
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"
  from_detection_checkpoint: true
  num_steps: 200000
}
train_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_train.record"
  }
}
eval_config {
  num_examples: 8000
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/mscoco_label_map.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/mscoco_val.record"
  }
}

#after
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "faster_rcnn_nas_coco/model.ckpt"
  from_detection_checkpoint: true
  num_steps: 200000
}
train_input_reader {
  label_map_path: "data/object-detection.pbtxt"
  tf_record_input_reader {
    input_path: "data/train.record"
  }
}
eval_config {
  num_examples: 8000
  max_evals: 10
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "data/object-detection.pbtxt"
  shuffle: false
  num_readers: 1
  tf_record_input_reader {
    input_path: "data/test.record"
  }
}
```

### 学習

```
python object_detection/model_main.py \
--logtostderr \
--model_dir=model \
--pipeline_config_path=pipeline.config
```

### 学習進捗を確認

```
tensorboard --logdir model

http://0.0.0.0:6006/
```

### 学習したcheckpointから推論用モデル（.pb形式）データを生成する

```
cd ./models/research/object_detection/

python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./faster_rcnn_nas_coco/pipeline.config  --trained_checkpoint_prefix ./tmplog/fast_rcnn_nas_coco/tmp6q8zwuzm/model.ckpt-3469  --output_directory  ./inference_graph/faster_rcnn_nas_coco/

# --input_type image_tensor  #固定
# --pipeline_config_path #学習時に指定したconfigファイル}
# --trained_checkpoint_prefix  # 学習時に出力されたckptデータ群指定　# model.ckpt-3469 #ここまで書けば良い
# --output_directory # 推論用データの出力先ディレクトリ　
```



以下ファイルが生成される


./inference_graph など適当なディレクトリを作ってそこに上記を全部入れておく
```
frozen_inference_graph.pb
model.ckpt.meta
model.ckpt.data-00000-of-00001
model.ckpt.index
checkpoint
pipeline.config
saved_model
```

注意）.pbファイルを生成した際と後の検出実行する際のtensorflowバージョンは合わせる必要ないとエラーになる。（今回は==1.13.1）



### 学習した.pbファイルで検出テスト

ここからこのモデルを使って他の画像を検出する 座標と画像が出力される

```
python3 from_root_room_detect_own_image.py  ./test.png
```


