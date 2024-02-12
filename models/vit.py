import tensorflow as tf
import keras
# import tensorflow_addons as tfa
# from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, concatenate, BatchNormalization, Dropout, Attention
from keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, CSVLogger
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Add, Activation, Lambda
from keras import backend as K
from keras.regularizers import l2, l1

# training settings
batch_size = 16          # number of images per batch
num_heads = 8          # nummber of attention heads in the multi-attention layer
num_epochs = 40           # number of training epochs
transformer_dropout = 0.25 # dropout rate of mlp inner of transformers
mlp_dropout = 0.25           # dropout rate of mlp outside of transformers
transformer_layers = 12   # number of stacked encoder layers
projection_dim = 64

image_size = 124
patch_size = 8
num_patches = (image_size // patch_size) ** 2

@tf.keras.utils.register_keras_serializable()
class Patches(layers.Layer):
    def __init__(self, patch_size,**kwargs):
        """Constructor"""
        self.patch_size = patch_size
        super(Patches, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["patch_size"] = self.patch_size
        return config
        
    def call(self, images):
        """Forward-pass function"""
        batch_size = tf.shape(images)[0]
        
        # pixel to patch conversion
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1,1,1,1],
            padding="VALID",
        )
        # get depth
        patch_dims = patches.shape[-1]
        
        # resize
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

@tf.keras.utils.register_keras_serializable()
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, **kwargs):
        """Constructor"""
        self.num_patches = num_patches                   
        self.projection = layers.Dense(units=64) # Dense layer that map the patched input to 64
        self.position_embedding = layers.Embedding(          # learnable embedding layer for the position (above)
            input_dim = num_patches, output_dim = 64
        )
        super(PatchEncoder, self).__init__(**kwargs)
    
    def get_config(self):
        config = super().get_config()
        config["num_patches"] = self.num_patches
        return config
        
    def call(self, patch):
        """Forward Pass"""
        # create a position matrix
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        
        # resize the input patch with Dense
        input_patch = self.projection(patch)
        
        # encode position array with learnable embedding layer
        pos_emb = self.position_embedding(positions)
        
        # patch + position embedding
        encoded_patch = input_patch + pos_emb
        
        return encoded_patch

def vit_model(trainX):
    # create a data augmentation layer
    data_aug = keras.Sequential(
        [
            layers.Normalization(),
            # layers.Resizing(IMAGE_SIZE, IMAGE_SIZE),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )

    # compute the mean and varaince of the training data for normalization
    data_aug.layers[0].adapt(trainX)

    # projection_dim = 64

    demo_in =Input(shape=(157,), name='Demo 1') 
    dense1 = Dense(157, activation='relu')(demo_in)
    dense2 = Dropout(0.25)(dense1)
    dense2 = Dense(128, activation='relu')(dense1)
    ehr_out = Dropout(0.25)(dense2)  


    input_shape = (124, 124, 3) # original input image size
    # set input layer
    inputs = layers.Input(shape=input_shape)
    # normalize and resize image
    augmented = data_aug(inputs)

    # convert image's pixels into patches
    patches = Patches(patch_size)(augmented)

    # encode patch by linearly transform patch with dense and add the learnable position encoder
    encoded_patches = PatchEncoder(num_patches)(patches)

    # att_dict = dict()
    # create stacked encoder
    for _ in range(transformer_layers):
        # layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        # multi-head attention
        mtha, attention_score = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1, return_attention_scores=True)

        # att_dict["transformer_att"+str(_)] = attention_score

        # skip connection 1 = add input with mtha
        x2 = layers.Add()([mtha, encoded_patches])

        # layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        # MLP
        x4 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x4 = layers.Dropout(transformer_dropout)(x4)
        x4 = layers.Dense(projection_dim, activation=tf.nn.gelu)(x4)
        x4 = layers.Dropout(transformer_dropout)(x4)

        # skip connection 2
        encoded_patches = layers.Add()([x4, x2])
    # flatten transformers
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(mlp_dropout)(representation)
    # features = layers.Dense(2048, activation=tf.nn.gelu)(representation)
    # features = layers.Dropout(mlp_dropout)(features)
    features = layers.Dense(1024, activation=tf.nn.gelu)(representation)
    features = layers.Dropout(mlp_dropout)(features)
    features = layers.Dense(512, activation=tf.nn.gelu)(features)
    features = layers.Dropout(mlp_dropout)(features)
    features = layers.Dense(256, activation=tf.nn.gelu)(features)
    features = layers.Dropout(mlp_dropout)(features)
    features = layers.Dense(128, activation=tf.nn.gelu)(features)
    ecg_out = layers.Dropout(mlp_dropout)(features)

    merge = Concatenate()([ehr_out, ecg_out])
    hidden = Dense(128, activation='relu',name='dense1')(merge)
    drop1 = Dropout(0.25)(hidden)
    hidden2 = Dense(64, activation='relu') (drop1)
    output = Dense(1, activation='sigmoid',name='output2')(hidden2)

    #Building the Model
    model = tf.keras.Model(inputs = [demo_in, inputs], outputs=output)

    # output = layers.Dense(1, activation='sigmoid')(features)

    # create model
    # model = keras.Model(inputs=inputs, outputs=output)
    opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    metrics = [
        'accuracy',
        keras.metrics.FalseNegatives(name="fn"),
        keras.metrics.FalsePositives(name="fp"),
        keras.metrics.TrueNegatives(name="tn"),
        keras.metrics.TruePositives(name="tp"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(curve= 'ROC'),
    ]

    model.compile(loss=keras.losses.BinaryFocalCrossentropy(alpha=0.25, gamma=2.0, from_logits=False),weighted_metrics = metrics,optimizer = opt)
    
    return model

