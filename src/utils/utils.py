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



#calculate PCA 

def calculate_pca(training_outputs):
    
    #create PCA object
    pca = PCA(n_components=10)
    
    #fit PCA to training outputs
    pca.fit(training_outputs)
    
    
    #transform training and test outputs
    training_outputs_pca = pca.transform(training_outputs)


    
    return training_outputs_pca, pca



def apply_pca_to_rois(training_outputs, roi_assignments, n_rois=7, n_components=10):
    # Initialize the output array
    pca_outputs = np.zeros((training_outputs.shape[0], n_rois * n_components))

    # Loop over each ROI
    for roi_index in range(n_rois):
        # Find the voxels belonging to the current ROI
        roi_voxels = np.where(roi_assignments == roi_index+1)[0]

        # Apply PCA on the current ROI across all samples
        pca = PCA(n_components=n_components)
        roi_data = training_outputs[:, roi_voxels]
        pca_transformed = pca.fit_transform(roi_data)

        # Place the PCA components into the output array
        start_col = roi_index * n_components
        end_col = start_col + n_components
        pca_outputs[:, start_col:end_col] = pca_transformed

    return pca_outputs