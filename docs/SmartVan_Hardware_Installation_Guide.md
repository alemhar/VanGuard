# SmartVan Monitor System
# Hardware Installation Guide

**Version 1.0.0**  
**Date: May 26, 2025**

## Table of Contents

1. [System Overview](#system-overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Camera Installation](#camera-installation)
4. [Computing Unit Setup](#computing-unit-setup)
5. [Power Configuration](#power-configuration)
6. [Network Configuration](#network-configuration)
7. [Software Installation](#software-installation)
8. [Downloading Detection Models](#downloading-detection-models)
9. [System Testing](#system-testing)
10. [Troubleshooting](#troubleshooting)

## System Overview

The SmartVan Monitor system provides intelligent inventory monitoring for delivery and service vehicles. Using computer vision technology, the system detects and logs inventory access events, helping to prevent theft and improve operational efficiency.

## Hardware Requirements

### Required Components
- **Computing Unit**: Raspberry Pi 4 (8GB RAM) or comparable embedded system
- **Cameras**: 2-4 wide-angle cameras with low-light capability (minimum 720p resolution)
- **Storage**: 128GB minimum high-endurance microSD card or SSD
- **Power Supply**: 12V to 5V converter with minimum 3A output
- **Network**: 4G/LTE modem for remote connectivity (optional)
- **Enclosure**: Vibration-resistant, temperature-resistant mounting case
- **Cables**: USB extension cables, power cables, camera mounts

### Tools Required
- Phillips screwdriver
- Wire stripper/crimper
- Cable ties
- Double-sided mounting tape or brackets
- Drill with appropriate bits (for mounting)

## Camera Installation

### Optimal Camera Placement

1. **Inventory Area Coverage**:
   - Position cameras to cover all inventory storage areas
   - Ensure full visibility of access points and shelving units
   - Avoid blind spots where items could be removed without detection

2. **Camera Mounting**:
   - Mount cameras at upper corners for maximum coverage
   - Secure cameras with the provided brackets
   - Ensure cameras are positioned to minimize vibration effects

3. **Camera Angle Guidelines**:
   - Primary inventory cameras: 45° downward angle recommended
   - Door monitoring cameras: Directly facing the entry point
   - Avoid pointing cameras at direct light sources

4. **Cable Management**:
   - Route cables along existing vehicle wiring paths where possible
   - Secure cables with cable ties every 30cm
   - Provide strain relief at connection points
   - Keep all cables at least 15cm from vehicle electrical components

### Camera Configuration

Based on our optimized settings:

- **Frame Rate**: Configure cameras for 2 FPS (500ms intervals)
- **Resolution**: 720p recommended (higher resolutions increase CPU load)
- **Focus**: Set to infinity focus for inventory areas
- **Exposure**: Auto-exposure with bias toward brighter images

## Computing Unit Setup

1. **Mounting Location**:
   - Install the computing unit in a vibration-isolated, ventilated location
   - Keep away from direct heat sources
   - Ensure accessibility for maintenance
   - Recommended locations: Under dashboard or in dedicated equipment area

2. **Connection Diagram**:
   ```
   [Camera 1] ---USB---> [Computing Unit] <---Power--- [Vehicle Power]
   [Camera 2] ---USB---> [Computing Unit] 
   [Camera 3] ---USB---> [Computing Unit] <---Network--- [4G Modem]
   [Camera 4] ---USB---> [Computing Unit]
   ```

3. **Secure Installation**:
   - Use provided vibration-dampening mounts
   - Secure all connections with locking mechanisms where available
   - Apply thread-lock to mounting screws

## Power Configuration

1. **Power Source Options**:
   - **Direct Battery Connection**: Connect to vehicle battery with proper fusing
   - **Accessory Power**: Connect to ignition-switched power source
   - **Hybrid Solution**: Main power from battery with ignition signal for sleep mode

2. **Power Management**:
   - Install the provided voltage regulator between vehicle power and computing unit
   - Configure automatic shutdown when vehicle is off (to prevent battery drain)
   - Implement graceful shutdown sequence on power loss

3. **Wiring Guidelines**:
   - Use 16 AWG wire minimum for power connections
   - Install a 5A fuse at the power source connection
   - Ensure all ground connections are secure and corrosion-free

## Network Configuration

1. **Local Network**:
   - Configure the computing unit as a local Wi-Fi access point
   - Default SSID: `SmartVan_[VehicleID]`
   - Default Password: Provided in sealed envelope

2. **Remote Connectivity** (if equipped with 4G/LTE):
   - Install the cellular modem in a location with good reception
   - Connect to computing unit via Ethernet or USB
   - Configure APN settings according to carrier requirements

## Software Installation

1. **Operating System**:
   - The system comes pre-installed with the required OS
   - If reinstallation is needed, follow the OS Reinstallation Guide

2. **SmartVan Software Installation**:
   - Login to the computing unit via SSH or direct connection
   - Clone the repository:
     ```bash
     git clone https://github.com/company/smartvan-monitor.git
     ```
   - Navigate to the installation directory:
     ```bash
     cd smartvan-monitor
     ```
   - Run the installation script:
     ```bash
     sudo ./install.sh
     ```

3. **Configuration**:
   - Edit the main configuration file:
     ```bash
     nano config/main.json
     ```
   - Set the vehicle ID, camera paths, and detection zones
   - Save and exit

## Downloading Detection Models

The SmartVan Monitor system uses YOLO (You Only Look Once) object detection for enhanced detection accuracy. These models must be downloaded before the system can operate at full capability.

### Automatic Model Download

1. **Connect to Internet**:
   - Ensure the computing unit has an active internet connection
   - Verify connectivity with `ping google.com`

2. **Download Models**:
   - Run the model downloader script:
     ```bash
     cd /path/to/smartvan-monitor
     python download_yolo_models.py
     ```
   - This will download the required models (may take 5-30 minutes depending on connection speed)
   - Progress will be displayed during download

3. **Verify Downloads**:
   - Check that models were downloaded successfully:
     ```bash
     ls -la detection/models/
     ```
   - You should see the following files:
     - `yolov4-tiny.cfg` (~12 KB)
     - `yolov4-tiny.weights` (~23 MB)
     - `yolov4.cfg` (~13 KB) 
     - `yolov4.weights` (~245 MB)

### Model Selection

The system is configured to use the "tiny" model by default for optimal performance. If higher detection accuracy is required:

1. **Edit Configuration**:
   ```bash
   nano config/main.json
   ```

2. **Change Model Size**:
   - Locate the object_detection section
   - Change `"model_size": "tiny"` to `"model_size": "medium"` or `"model_size": "large"`
   - Note: Larger models consume more resources but provide higher accuracy

3. **Save and Restart**:
   ```bash
   sudo systemctl restart smartvan.service
   ```

## System Testing

After installation, perform these tests to verify system operation:

### 1. Camera Functionality Test
- Run `python -m tools.camera_test`
- Verify all cameras are producing clear images

### 2. Motion Detection Test
- Run `python -m tools.motion_test`
- Move in front of each camera to verify motion detection
- Verify settings match our optimized configuration:
  - flow_magnitude_threshold: 3.5
  - flow_uniformity_threshold: 0.5
  - significant_motion_threshold: 6.0
  - min_duration: 1.5

### 3. Inventory Detection Test
- Run `python -m tools.inventory_test`
- Open inventory compartments and remove/add items
- Verify the system detects and classifies events correctly

### 4. Network Connectivity Test
- Run `python -m tools.network_test`
- Verify the system can connect to the backend services

### 5. Power Management Test
- Run `python -m tools.power_test`
- Verify system enters low-power mode when vehicle ignition is off
- Verify clean shutdown when power is removed

## Troubleshooting

### Camera Issues
- **No image**: Check USB connections, try different USB port
- **Dark image**: Adjust camera exposure settings
- **Blurry image**: Clean lens, check focus setting

### Detection Issues
- **False positives**: Adjust motion thresholds in config file
- **Missed events**: Lower detection thresholds, check camera positioning
- **System overheating**: Check ventilation, consider additional cooling

### Network Issues
- **No connectivity**: Check modem power, verify SIM activation
- **Intermittent connection**: Check antenna placement, signal strength

### Power Issues
- **System shuts down while driving**: Check power connections, verify adequate current supply
- **Battery drain**: Verify sleep mode is working, check power management settings

## Contact Information

For installation support:
- Technical Helpdesk: (555) 123-4567
- Email: support@smartvanmonitor.com
- Hours: Monday-Friday, 8AM-8PM EST

---

*This document is confidential and proprietary. Unauthorized reproduction or distribution is prohibited.*
