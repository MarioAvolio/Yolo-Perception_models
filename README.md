# Yolo-DinoV2
Yolo-DinoV2 integrates Meta's DINOv2, a self-supervised Vision Transformer model, as a frozen feature extractor backbone into Ultralytics' YOLO object detection framework. This combination aims to enhance few-shot object detection capabilities, allowing for effective training with minimal data.

## Features

- **⭐ DINOv2 Backbone**: Utilizes DINOv2's robust feature extraction to improve object detection performance. All DinoV2 backbone sizes supported (uses registers)

- **⭐ Seamless Integration**: Maintains compatibility with Ultralytics' YOLO, enabling standard training and inference workflows.

- **⭐ Few-Shot Learning**: Designed to generalize well with relatively small datasets, facilitating quick training on custom data.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/itsprakhar/Yolo-DinoV2.git
   cd Yolo-DinoV2
   ```

2. **Install Dependencies**:

   Ensure you have Python installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **⚙️ Model Initialization**:

   Use the configuration files in the `yolo_dinov2_configs` directory to initialize the YOLO model with the DINOv2 backbone. Modify the `nc` parameter in the YAML file to match the number of classes in your dataset.

2. **🏋️ Training**:

   Train the model as you would with any YOLO model from Ultralytics. Refer to Ultralytics' [training documentation](https://docs.ultralytics.com/tasks/detect/#train) for detailed instructions.

3. **🏃Inference**:

   Same  you would with any YOLO model from Ultralytics. Refer to Ultralytics' [Inference documentation](https://docs.ultralytics.com/tasks/detect/#predict) for detailed instructions.


- **🥹 Pretrained Weights**: Pretrained weights are not available. However, the model can be trained quickly on custom data with relatively small datasets 😊, thanks to its strong generalization capabilities. 

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the functionality of this project.


## Acknowledgements

- **DINOv2 🦕**: Developed by Meta AI Research, DINOv2 is a self-supervised Vision Transformer model that produces high-performance visual features applicable across various computer vision tasks. 

- **Ultralytics YOLO 🚀**: A state-of-the-art real-time object detection model known for its speed and accuracy. 


<hr>



>**If you find this project useful, please consider giving it a ⭐ on GitHub to motivate further development.**

Enjoy using Yolo-DinoV2 for your object detection tasks! 🤩