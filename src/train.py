import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model import create_model

def train_model(data_dir, model_save_path, input_shape=(128, 128, 3), batch_size=32, epochs=10):
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    model = create_model(input_shape, num_classes=len(train_generator.class_indices))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs
    )
    
    model.save(model_save_path)
    return model
