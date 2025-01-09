# CubeSat_pytorch
 Repository for the development of a neural network that recognises synthetic CubeSats, cubit satellites generated using Unreal Engine (see https://www.kaggle.com/datasets/eberhardtkorf/synthetic-cubesat). The dataset contains 7998 training images, and 2000 testing images. Furthermore, in this analysis the training set is divided into a 6670 set for actual training, and a 1328 set for validation.
 
 The object recognition is performed using a **convolutional neural network (CNN)**, as is commonly done for computer vision tasks, which alternates convolutional layers, max-pooling layers and relu activations with 2D dropout, the output of these being passed to a series of linear layers which have dropout as well.

 The task has been taken on as a **multi-label regression** task, each label being one of the coordinates of the object. The problem has been split into two networks: one which predicts the (x, y) position normalised by the pixel dimensions of the image (which has been reduced to a lower resolution due to limited computational resources), and another one for the perpendicular distance from the camera, z. This choice of separating the problem is motivated by the fact that the (x,y) coordinates are normalised, while the z is not (see the end of this text for more on this matter). The projection of the true position onto the (x, y) pixel coordinates has been done using the specifications of the (simulated) camera used, via a camera matrix and the *OpenCV library*.
 
 The structure of both convolutional networks is the same, only varying in the number of outputs. Using the convention for the dimensions as Height x Width (in the same order as pytorch's kernel and stride parameters):

 - 3 -> 64 channels 8x8 kernel **Conv2d layer** using 4x2 stride, with **2D dropout** connections to a 3x2 kernel **MaxPool2d layer** using 2x2 stride. The output of this layer is passed through the **ReLU activation** function.
 - 64 -> 128 channels 4x4 kernel **Conv2d layer** using 1x1 stride, connected to a 2x2 kernel **MaxPool2d layer** layer using 2x2 stride. The output of this layer is passed through the **ReLU activation** function.
 - 128 -> 256 channels 3x4 kernel **Conv2d layer** using 1x2 stride, connected to a 2x3 kernel **MaxPool2d layer** layer using 2x2 stride. The output of this layer is passed through the **ReLU activation** function and **flattened** for the next layer.
 - 12288 (256 x 6 x 8) neurons with **dropout** connections to the previous layer, and fully connected to num_labels output units.

The num_labels is to be chosen by the user. The loss function used is **Huber loss**, which depending on the error uses L1 or Mean Squared Error loss, making it more robust to outliers. The total loss is the mean value of the losses across all elements in the current batch. Other specifications for training are:

- Usage of an **Adam optimizer**, with initial **learning rate of 0.001**
- **Learning rate scheduler**, which reduces the learning rate by a factor of 0.1 when the validation loss plateaus (does not decrease) for 4 epochs
- **Weight decay** regularization, with lambda parameter of 1e-4
- Early stop with a patience of 6 epochs
- Use of **automatic mixed precision (amp)** for improved computational efficiency

Furthermore, the training function allows for a few more functionalities if desired by the user, like the use of SGD and momentum-based SGD. The code uses a GPU if available. The low patience (or im-patience) of the model is due to the high use of computational resources, which slows down training. While this project is STILL UNDER DEVELOPMENT, in the near future tests will be carried out to check whether or not a single network works well for regressing all 3 (x,y,z) labels.
