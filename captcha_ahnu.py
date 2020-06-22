import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers, callbacks, models
import os
import glob


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 设置显存按需分配
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

strategy = tf.distribute.MirroredStrategy()   # 用于分布式训练

IMAGES_PATH = 'images'
BATCH_SIZE = 256 * strategy.num_replicas_in_sync    # BATCH_SIZE为256*分布式设备数
CAPTCHA_CHARACTERS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']    # 验证码字符集
CAPTCHA_CHARACTERS_LENGTH = len(CAPTCHA_CHARACTERS)
CAPTCHA_LENGTH = 4    # 验证码位数
MODEL_INPUT_WIDTH = 160    # 网络输入宽度
MODEL_INPUT_HEIGHT = 32    # 网络输入高度
MODEL_INPUT_CHANNALS = 1    # 网络输入通道数

img_w = 150
img_h = 30
lr = 1e-3
epochs = 30
save_path = 'captcha_ahnu.h5'    # 模型保存文件名
checkpoint_path = 'checkpoint'    # 检查点路径
if not os.path.exists(checkpoint_path):
    os.mkdir(checkpoint_path)
checkpoint_filepath = os.path.join(checkpoint_path, 'capt_ahnu_weights.{epoch:02d}-{val_loss:.2f}-{val_categorical_accuracy:.2f}.hdf5')    # 检查点保存文件名格式为capt_ahnu_weights+迭代次数+验证集loss+验证集准确率
log_dir = 'tf_logs'    # tensorboard 日志路径



def load_img(image_file):
    '''
    加载图片
    '''
    image = tf.io.read_file(image_file)
    # image = tf.image.decode_image(image, channels=0)    # 多通道
    image = tf.image.decode_image(image, channels=1)    # 单通道
    image = tf.cast(image, tf.float32)
    return image

def normalize(image):
    '''
    像素值归一化到 -1~1
    '''
    image = (image / 127.5) - 1

    return image


def load_image_train(imgs, labels):
    '''
    加载一批次训练数据
    '''
    image = load_img(imgs)
    image = normalize(image)
    image = tf.image.resize(image, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH])
    
    label = tf.strings.bytes_split(labels)
    
    label = tf.strings.to_number(label, tf.int32)
    label = tf.one_hot(label, depth=10)
    label = tf.cast(label, tf.float32)
    
    return image, label

@tf.autograph.experimental.do_not_convert
def load_image_train_wrapper(imgs, labels):
    result_tensors = tf.py_function(load_image_train, [imgs, labels], [tf.float32, tf.float32])
    result_tensors[0].set_shape([MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 1])
    result_tensors[1].set_shape([CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH])
    return result_tensors

# 加载全部数据
all_imgs = glob.glob(IMAGES_PATH + '/*.png')
all_labels = [os.path.splitext(os.path.split(i)[1])[0][-CAPTCHA_LENGTH:] for i in all_imgs]
all_imgs = tf.convert_to_tensor(all_imgs)
all_labels = tf.convert_to_tensor(all_labels)



# 划分训练验证集比例
train_num = int(len(all_labels) * 0.95)
assert train_num < len(all_labels) and len(all_labels) > 0
train_imgs, test_imgs = all_imgs[:train_num], all_imgs[train_num:]
train_labels, test_labels = all_labels[:train_num], all_labels[train_num:]

train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
train_db = train_dataset.map(load_image_train_wrapper).batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_imgs, test_labels))
test_db = test_dataset.map(load_image_train_wrapper).batch(BATCH_SIZE)


# 定义模型函数，更深会更慢，经测试四层卷积已能达到100%准确率
def make_model():
    
    input_size = (MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, MODEL_INPUT_CHANNALS)
    input_layer = layers.Input(shape=input_size)
    
    conv_1_1 = layers.Conv2D(16, kernel_size=3, strides=(1, 1), padding='same')(input_layer)
    bn_1_1 = layers.BatchNormalization()(conv_1_1)
    relu_1_1 = layers.LeakyReLU()(bn_1_1)
    conv_1_2 = layers.Conv2D(32, kernel_size=3, strides=(2, 2), padding='same')(relu_1_1)
    bn_1_2 = layers.BatchNormalization()(conv_1_2)
    relu_1_2 = layers.LeakyReLU()(bn_1_2)
    conv_out = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(relu_1_2)
    
    conv_2_1 = layers.Conv2D(64, kernel_size=3, strides=(1, 1), padding='same')(conv_out)
    bn_2_1 = layers.BatchNormalization()(conv_2_1)
    relu_2_1 = layers.LeakyReLU()(bn_2_1)
    conv_2_2 = layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same')(relu_2_1)
    bn_2_2 = layers.BatchNormalization()(conv_2_2)
    relu_2_2 = layers.LeakyReLU()(bn_2_2)
    conv_out = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(relu_2_2)
    
    '''
    conv_3_1 = layers.Conv2D(128, kernel_size=3, strides=(1, 1), padding='same')(conv_out)
    bn_3_1 = layers.BatchNormalization()(conv_3_1)
    relu_3_1 = layers.LeakyReLU()(bn_3_1)
    conv_3_2 = layers.Conv2D(256, kernel_size=3, strides=(1, 1), padding='same')(relu_3_1)
    bn_3_2 = layers.BatchNormalization()(conv_3_2)
    relu_3_2 = layers.LeakyReLU()(bn_3_2)
    conv_out = layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same')(relu_3_2)
    '''
    

    flatten = layers.Flatten()(conv_out)
    fc1 = layers.Dense(128)(flatten)
    bn_4 = layers.BatchNormalization()(fc1)
    relu_4_1 = layers.LeakyReLU()(bn_4)
    
    
    # 全连接层，输出通道数为分类数
    final_fc = layers.Dense(CAPTCHA_CHARACTERS_LENGTH * CAPTCHA_LENGTH)(relu_4_1)
    
    outputs_logits = layers.Reshape((CAPTCHA_LENGTH, CAPTCHA_CHARACTERS_LENGTH))(final_fc)
    
    outputs = layers.Softmax()(outputs_logits)
    
    # 定义模型，指定输入与输出
    model = Model(inputs=[input_layer], outputs=[outputs])
    return model


# 训练
def train():
    # 定义优化器
    opt = optimizers.Adam(lr)
    # 定义回调函数，包含防止过拟合、日志统计、自动保存模型等
    callback = [callbacks.TensorBoard(log_dir=log_dir, update_freq='batch'),
            callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3),
            callbacks.EarlyStopping(monitor='loss', patience=4),
            callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                      verbose=1),
            callbacks.ModelCheckpoint(filepath=save_path,
                                      monitor='val_categorical_accuracy', 
                                      save_best_only=True,
                                      mode='max',
                                      verbose=1)
            ]
    
    # 分布式训练
    with strategy.scope():
        model = make_model()
        if os.path.exists(save_path):
            model.load_weights(save_path)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(train_db, epochs=epochs, validation_data=test_db, callbacks=callback)    # 开始训练
    model.evaluate(test_db)    # 在测试集中评估模型
    model.save(save_path)    # 保存最终模型
    
    
# 测试
def test():
    if os.path.exists(save_path):

        model = models.load_model(save_path)
        model.evaluate(test_db)
    else:
        print('faild to load a model from "%s"' % (save_path))
       
    
# 解码onehot为标签
def decode_labels(labels):
    index = tf.math.argmax(labels, axis=-1)
    ret = [''.join([str(j) for j in i]) for i in index.numpy()]
    return ret
    
    
# 计算准确率
def compute_acc(labels, pred_labels):
    real = decode_labels(labels)
    pred = decode_labels(pred_labels)
    assert len(real) == len(pred)
    
    correct = 0
    for i in range(len(real)):
        if real[i] == pred[i]:
            correct += 1
        print('real: %s,\tpred: %s' % (real[i], pred[i]))
    acc = correct / len(real)
    return acc
    
    
        
# 测试检测效果
def detect_test():
    if os.path.exists(save_path):

        model = models.load_model(save_path)
        imgs, labels = next(iter(test_db))
        pred_labels = model(imgs)
        acc = compute_acc(labels, pred_labels)
        print('acc: %.2f' % acc)
    else:
        print('faild to load a model from "%s"' % (save_path))
        
# main
if __name__ == '__main__':
    # train()    
    # test()    
    detect_test()


