import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import tensorflow_hub as hub
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse
import matplotlib.pyplot as plt 
import os
parser=argparse.ArgumentParser()
parser.add_argument("--batch-size",type=int,default=10)
args =parser.parse_args()
def main():
    train_ds  = tf.keras.utils.image_dataset_from_directory(
        "data/train",
        batch_size=args.batch_size,
        image_size=(224, 224),
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        "data/valid",
        batch_size=args.batch_size,
        image_size=(224, 224),

    )

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))


    # Setting optimizer
    optimizer=optimizers.Adam(lr=0.005)
    decay_steps = 20
    initial_learning_rate = 0
    warmup_steps = 20
    target_learning_rate = 0.1
    lr_warmup_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
        warmup_steps=warmup_steps
)   

    # Setting model
    model = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/tensorflow/efficientnet/b0/classification/1",
                    trainable=True)
    ])
    model.build([None, 224, 224, 3])  # Batch input shape.

    model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])
    #Training step
    checkpoint_path = "weights/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)



    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)


    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                    save_weights_only=True,
                                                    verbose=1,
                                                    save_freq=1)


    history_fine=model.fit(
    train_ds,
    validation_data=val_ds,
    epochs= 10,callbacks=[cp_callback,early_stopping,reduce_lr]
    )



    ## Training results
    plt.figure()
    plt.plot(history_fine.history["accuracy"])
    plt.plot(history_fine.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train","validation"],loc="upper left")
    plt.savefig("training_history.png")
    return

if __name__=="__main__":
    main()
