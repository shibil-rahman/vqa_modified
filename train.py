from tensorflow.keras.callbacks import ModelCheckpoint
import argparse
from model import build_model
from prepare_data import setup

# Prepare data
train_X_ims, train_X_seqs, train_Y, test_X_ims, test_X_seqs, test_Y, im_shape, vocab_size, num_answers, _, _, _ = setup()

print('\n--- Building model...')
model = build_model(im_shape, vocab_size, num_answers, args.big_model)
checkpoint = ModelCheckpoint('model.h5', save_best_only=True)

print('\n--- Training model...')
print("checkpoint==",checkpoint)
model.fit(
  [train_X_ims, train_X_seqs],
  train_Y,
  validation_data=([test_X_ims, test_X_seqs], test_Y),
  shuffle=True,
  epochs=1,
  callbacks=[checkpoint],
)

from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import confusion_matrix

y_pred = model.predict([test_X_ims, test_X_seqs])
y_pred=np.argmax(y_pred, axis=1)
y_test=np.argmax(test_Y, axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.3f' % (accuracy * 100))
