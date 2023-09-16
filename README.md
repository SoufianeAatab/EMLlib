# EMLlib - Embedded Systems Machine Learning Library

EMLlib is a C/C++ machine learning library designed for embedded systems. It enables developers to integrate machine learning capabilities into resource-constrained microcontrollers and embedded devices. This README provides instructions for getting started with EMLlib in your development environment.

## Prerequisites

Before you begin, ensure you have the following prerequisites:

- [Visual Studio Code](https://code.visualstudio.com/) installed on your computer.
- [PlatformIO](https://platformio.org/) extension installed within Visual Studio Code.

## Installation

1. **Clone the EMLlib repository:** Clone the EMLlib repository to your local machine using your preferred Git client or by running the following command in your terminal:

   ```
   git clone https://github.com/SoufianeAatab/EMLlib.git
   ```

2. **Open Visual Studio Code:** Launch Visual Studio Code.

3. **Add Existing Project:**

   - Go to the PlatformIO Home by clicking on the PlatformIO icon in the left sidebar.
   - Click on the "Projects" button.

4. **Add EMLlib Project:**

   - Click on "Add Existing" and browse to the location where you cloned the EMLlib repository.
   - Select the `examples/feed-forward` project within the EMLlib repository.

## Building the Project

Once you've added the EMLlib project to your PlatformIO workspace, you can build it as follows:

1. In Visual Studio Code, navigate to the PlatformIO Home by clicking on the PlatformIO icon in the left sidebar.

2. Click on the "Projects" button to see your added projects.

3. Select the `examples/feed-forward` project.

4. Build the project by clicking on the build button (usually represented as a checkmark icon) in the PlatformIO toolbar.

## Uploading to Your Target Device

EMLlib supports a variety of embedded platforms. To upload the project to your specific target device, follow these steps:

1. Connect your target embedded device (e.g., Arduino Portenta H7) to your computer.

2. In Visual Studio Code, ensure that the correct target device is selected in the PlatformIO toolbar.

3. Click on the upload button in the PlatformIO toolbar. This will compile and upload the project to your connected device.

## License

This project is licensed under the [MIT License](LICENSE). Feel free to modify and use it for your own embedded machine learning applications.
