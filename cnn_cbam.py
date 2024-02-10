import tensorflow as tf
import keras
# import tensorflow_addons as tfa
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate, BatchNormalization, Dropout, Attention
from keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Add, Activation, Lambda
from keras import backend as K
from keras.regularizers import l2, l1

def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature

def channel_attention(input_feature, ratio=8):
    
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    
    shared_layer_one = Dense(channel//ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    avg_pool = GlobalAveragePooling2D()(input_feature)    
    avg_pool = Reshape((1,1,channel))(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    avg_pool = shared_layer_one(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel//ratio)
    avg_pool = shared_layer_two(avg_pool)
    assert avg_pool.shape[1:] == (1,1,channel)
    
    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    max_pool = shared_layer_one(max_pool)
    assert max_pool.shape[1:] == (1,1,channel//ratio)
    max_pool = shared_layer_two(max_pool)
    assert max_pool.shape[1:] == (1,1,channel)
    
    cbam_feature = Add()([avg_pool,max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
    kernel_size = 7
    
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2,3,1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(concat)	
    assert cbam_feature.shape[-1] == 1
    
    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)
    
    return multiply([input_feature, cbam_feature])

def buildModel(n_features=157):

    """The branch dealing with the EHR Data"""
    demo_in =Input(shape=(n_features,), name='Demo 1') 
    dense1 = Dense(n_features, activation='relu')(demo_in)
    dense2 = Dropout(0.25)(dense1)
    dense2 = Dense(128, activation='relu')(dense2)
    ehr_out = Dropout(0.25)(dense2)    

    """The branch dealing with ECG Data"""
    lead1 = Input(shape=(124, 124, 3))
    x = Conv2D(64, (3, 3), padding="same", activation='relu')(lead1)
    x = cbam_block(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = cbam_block(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)
    x1 = Conv2D(128, (3, 3), padding="same", activation='relu')(x)

    lead2 = Input(shape=(124, 124, 3))
    x = Conv2D(64, (3, 3), padding="same", activation='relu')(lead2)
    x = cbam_block(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)
    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = cbam_block(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)
    x2 = Conv2D(128, (3, 3), padding="same", activation='relu')(x)

    concatenated = concatenate([x1, x2])
    attn = cbam_block(concatenated)
    x = Dropout(0.5)(attn) 
    x = Conv2D(256, (3, 3), padding="same", activation='relu')(x)
    x = cbam_block(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)
    x = Dropout(0.5)(x) 
    x = Conv2D(256, (3, 3), padding="same", activation='relu')(x)
    x = cbam_block(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((3, 3))(x)
    flatten = Flatten()(x)
    x = Dense(1024, activation='relu')(flatten)
    ecg_out = Dense(512, activation='relu')(x)

    """Combining both the branches"""
    merge = Concatenate()([ehr_out,ecg_out])
    hidden = Dense(128, activation='relu',name='dense1')(merge)
    drop1 = Dropout(0.25)(hidden)
    hidden2 = Dense(1, activation='sigmoid',name='output2')(drop1)

    #Building the Model
    model = tf.keras.Model(inputs = [demo_in, lead1, lead2], outputs=hidden2)

    print("Compiling the Model with optimizers and Metrics")
    opt = tf.keras.optimizers.Adam(learning_rate = 1e-4)
    weighted_metrics = [
        'accuracy',
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(curve= 'ROC'),
    ]

    model.compile(loss=keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0, from_logits=False)
                ,weighted_metrics = weighted_metrics,optimizer = opt)

    return model