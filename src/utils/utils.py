from .imports import *

# Transform to C x H x W = 3 x 224 x 224 by resizing and duplicating the grayscale channel
def preprocess_images(data):
    transformed_images = []

    # Define the transformations to be applied to each image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for i in range(data.shape[0]):
        # Convert numpy array to PIL Image
        image = Image.fromarray(np.uint8(data[i] * 255), 'L')

        # Apply the transformations
        transformed_image = preprocess(image)
        transformed_images.append(transformed_image)

    # Stack all the transformed images into a single tensor
    tensor_stack = torch.stack(transformed_images)

    # Convert the tensor to a numpy array
    numpy_array = tensor_stack.numpy()

    return numpy_array

def get_current_time():
    # Saving model - using time as unique identifier for name
    t = time.localtime()
    current_time = time.strftime("%D_%H_%M_%S", t)

    #make the day format have a _ instead of /
    current_time = current_time.replace("/","_")

    #time as string
    timeString = str(current_time)

    return timeString