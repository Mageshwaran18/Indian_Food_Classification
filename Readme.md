## Flow Chart

### 1. Create an Environment

--> To create an enviironment, run the following command in the terminal: conda create -n food-classification python=3.9
--> To activate the enviornment, run the following command in the terminal: conda activate food-classification
--> Install the required packages by running the following command in the terminal: pip install -r requirements.txt

### 2. Data Collection
- Number of Classes = 7
  - Dosa
  - Idly
  - Noodles
  - Purrota
  - Grill Chicken
  - Fish Fry
Use this extension to download images from the google -> https://chromewebstore.google.com/detail/download-all-images/ifipmflagepipjokmbdecpmjbibjnakm?hl=en

### 3. Data Preprocessing
- Resize images to 224 x 224 (Input shape for MobileNet)
- Normalize pixel values between 0 and 1
- Apply data augmentation --> Using ImageDataGenerator class from keras.preprocessing.image
- Handle class imbalance --> Stratified K fold cross validation

### 4. Model Development
- Design the architecture of the model
- Train the model
- Perform transfer learning
- Compare the two models
- Save and load the model

### 5. Deployment
- Create a Flask application
- Deploy to AWS