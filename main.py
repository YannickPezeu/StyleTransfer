# SETUP AND IMPORTS

import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

import sys


# DEFINE CONSTANTS
MAX_DIM = 512 # Dimension of the longest side of the image
TOTAL_VARIATION_WEIGHT = 0
EPOCHS = 45
STEPS_PER_EPOCHS = 20
if len(sys.argv) > 2:
    content_path = "origins/content/" + sys.argv[2]
else:
    content_path = "origins/content/photoCV haute resolution_cropped.png"

if len(sys.argv) > 1:
    style_path = sys.argv[1]
else:
    style_path = "Monet2-.jpg"

if "origins" in style_path:
    style_path = style_path[8:] # Remove origins/ from style_path
origin_style_path = 'origins/styles/' + style_path
output_style_path = 'results/' + style_path
print("STYLE_PATH")
print(style_path)

# CONFIGURE GPUS
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)




def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

# VISUALIZE

def load_img(path_to_img):
  max_dim = MAX_DIM
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

  # plt.show()



content_image = load_img(content_path)
style_image = load_img(origin_style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')


content_layers = ['block5_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

## Build the Model

def vgg_layers(layer_names): ## Returns intermediate layers output
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

#Look at the statistics of each layer's output
for name, output in zip(style_layers, style_outputs):
  print(name)
  print("  shape: ", output.numpy().shape)
  print("  min: ", output.numpy().min())
  print("  max: ", output.numpy().max())
  print("  mean: ", output.numpy().mean())
  print()

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}


extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

print('styles:')
for name, output in sorted(results['style'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())
  print()

print("Contents:")
for name, output in sorted(results['content'].items()):
  print("  ", name)
  print("    shape: ", output.numpy().shape)
  print("    min: ", output.numpy().min())
  print("    max: ", output.numpy().max())
  print("    mean: ", output.numpy().mean())

  style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

image = tf.Variable(content_image)

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight=1e-2
content_weight=1e4

def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step_backup(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))





import time
start = time.time()

epochs = EPOCHS
steps_per_epoch = STEPS_PER_EPOCHS


####### TOTAL VARIATION LOSS

def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var

x_deltas, y_deltas = high_pass_x_y(content_image)





def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

total_variation_weight=TOTAL_VARIATION_WEIGHT

@tf.function()
def train_step(image):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))






step = 0

image_int = tf.cast(tf.squeeze(image)*255, tf.uint8)
content_path_base = os.path.basename(content_path)
tf.keras.utils.save_img(output_style_path[:-4]+str(0)+"_"+content_path_base+".jpeg", image_int)


steps_per_epoch = 0
for n in range(epochs):
  if n%3 == 0:
      steps_per_epoch += 1
  for m in range(steps_per_epoch):
    step += 1
    train_step(image)
    print(".", end='', flush=True)
  display.clear_output(wait=True)
  display.display(tensor_to_image(image))
  print("Train step: {}".format(step))
  image_int = tf.cast(tf.squeeze(image)*255, tf.uint8)
  content_path_base = os.path.basename(content_path)
  tf.keras.utils.save_img(output_style_path[:-4]+str(n)+"_"+content_path_base+"total_variation_loss_"+ str(TOTAL_VARIATION_WEIGHT)+".jpeg", image_int)
end = time.time()
print("Total time: {:.1f}".format(end-start))

imshow(image)
# plt.show()

import imageio
import os
from os import listdir
from os.path import isfile, join
from pathlib import Path


def create_gif(style_name, content_name):
    print("create_gif")
    mypath = "results"
    paths = sorted(Path(mypath).iterdir(), key=os.path.getmtime)
    print(type(paths))
    print(type([1,2]))
    onlyfiles = [ f for f in paths if (isfile(f) and ((style_name in str(f)) and (content_name in str(f)))) ]

    images = []
    for filename in onlyfiles:
        images.append(imageio.imread(filename))

    last_image = images[-1]
    for i in range(len(images)):
        images.append(last_image)
    try:
        os.mkdir('results_gif')
    except:
        print('can\'t create dir')
    imageio.mimsave('results_gif/'+ style_name + '_' + content_name + "total_variation_loss_"+ str(TOTAL_VARIATION_WEIGHT)+'.gif', images)
    print('results_gif/'+ style_name + '_' + "total_variation_loss_"+ str(TOTAL_VARIATION_WEIGHT)+ content_name + '.gif')


content_path_base = os.path.basename(content_path)
create_gif(style_path[0:-4], content_path_base[0:-4])
