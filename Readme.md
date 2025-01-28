## Flow Chart

### 1. Create an Environment and Setup GPU 

### 2. Data Collection
- Number of Classes = 7
  - Dosa
  - Idly
  - Noodles
  - Purrota
  - Chapathi
  - Grill Chicken
  - Fish Fry

Note: All classes except Grill Chicken and Fish Fry should have 200-250 data samples. Grill Chicken and Fish Fry will have class imbalance.

### 3. Data Preprocessing
- Resize images to 224 x 224 (Input shape for MobileNet)
- Normalize pixel values
- Apply data augmentation
- Handle class imbalance

### 4. Model Development
- Design the architecture of the model
- Train the model
- Perform transfer learning
- Compare the two models
- Save and load the model

### 5. Deployment
- Create a Flask application
- Deploy to AWS