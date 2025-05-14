# SmartVan Monitor: Detection Capabilities Technical Specification

This document provides detailed technical specifications for implementing the detection capabilities of the SmartVan Monitor system across all three development phases.

## Phase 1: MVP Detection Capabilities

### 1. Basic Motion Detection

**Technical Implementation:**
- **Algorithm**: Background subtraction using MOG2 or KNN background subtractor
- **Frame Rate**: Process at 5-10 FPS to conserve resources
- **Region Definition**: Define ROIs (Regions of Interest) for inventory areas
- **Sensitivity Tuning**: Configurable threshold to account for vehicle movement
- **Filtering**: Minimum size filter to ignore small movements (dust, shadows)

**Expected Outputs:**
- Motion bounding boxes
- Motion intensity score (0-100)
- Duration of continuous motion

**Edge Cases to Handle:**
- Vehicle movement (driving, stopping)
- Lighting changes (sun/shade, day/night transitions)
- Camera vibration from door closing

**Dependencies:**
- OpenCV motion detection libraries
- Motion mask generation

### 2. Person Presence Detection

**Technical Implementation:**
- **Model**: MobileNet SSD v2 or YOLOv4-tiny
- **Resolution**: 416x416 input resolution
- **Confidence Threshold**: >0.6 for person detection
- **Optimization**: TensorFlow Lite quantization (int8)

**Expected Outputs:**
- Person bounding boxes
- Detection confidence scores
- Person count in frame
- Rough position (front, middle, rear of van)

**Edge Cases to Handle:**
- Partial occlusion
- Multiple people
- Person wearing bulky clothing/uniform
- Low light conditions

**Dependencies:**
- Pre-trained person detection model
- TensorFlow Lite runtime

### 3. Inventory Access Event Detection

**Technical Implementation:**
- **Definition**: Person detected in proximity to inventory area + motion in inventory area
- **Proximity Calculation**: Overlap of person bounding box with predefined inventory zones
- **Event Start**: When person first enters inventory zone
- **Event End**: When person exits inventory zone or no motion for >15 seconds
- **Zone Mapping**: JSON configuration file defining inventory zones in pixel coordinates

**Expected Outputs:**
- Access event metadata (start time, end time, duration)
- Inventory zone identifier
- Confidence score based on visibility and motion intensity
- Screenshot of access initiation

**Edge Cases to Handle:**
- Person standing near but not accessing inventory
- Multiple people accessing inventory simultaneously
- Inventory access without clearly visible person

**Dependencies:**
- Person detection
- Motion detection
- Zone configuration system

### 4. Simple Temporal Patterns

**Technical Implementation:**
- **Time-window Analysis**: Sliding 15-minute window for event frequency
- **Access Counting**: Count access events per zone per time window
- **Frequency Threshold**: Flag when access count > 3 per zone within timeframe
- **Time-based Rules**: Implement rule checks based on local time

**Expected Outputs:**
- Frequency metrics per zone
- Time-stamped alerts when thresholds exceeded
- Historical access pattern data

**Edge Cases to Handle:**
- Normal high-frequency access during restocking
- Multiple short accesses vs. one long access
- Clock synchronization issues

**Dependencies:**
- Event database with timestamp indexing
- Time series analysis functions

## Phase 2: Enhanced Detection Capabilities

### 1. Box Opening Detection

**Technical Implementation:**
- **Approach 1**: Pose-based detection using MediaPipe or BlazePose
  - Track specific arm/hand movements associated with box opening
  - Detect characteristic postures
  
- **Approach 2**: Object-based detection
  - Detect boxes in closed state
  - Detect state change to open box
  - Track box flaps/lids
  
- **Model Architecture**: Custom-trained CNN or transformer-based model
- **Training Data Requirements**: 500+ labeled examples of box opening in van environment

**Expected Outputs:**
- Box opening events with confidence scores
- Box identifier/location
- Duration of box open state
- Before/after images

**Edge Cases to Handle:**
- Partial visibility of box opening
- Different box types and sizes
- Quick opening/closing actions
- Non-standard opening methods

**Dependencies:**
- Extended training data collection period
- Custom model training pipeline
- High-quality camera positioning

### 2. Item Removal Tracking

**Technical Implementation:**
- **Object Detection Base**: YOLOv5 or EfficientDet-Lite
- **Change Detection**: Compare inventory areas before/after access
- **Item Identification**: Train models on common inventory items
- **Tracking Method**: Either:
  - Direct item tracking when visible
  - Indirect (hand movement followed by item appearance)

**Expected Outputs:**
- Item removal events
- Item category/type if identifiable
- Approximate size/count of removed items
- Confidence score for removal detection

**Edge Cases to Handle:**
- Items remaining in hand vs. placed elsewhere in van
- Multiple similar items
- Partially visible removal
- Items returned to different location

**Dependencies:**
- Product catalog for training
- Inventory location mapping
- Fine-tuned object detection model

### 3. Staff Identification (Anonymized)

**Technical Implementation:**
- **Approach**: Feature-based identification without facial recognition
- **Features**: Height, build, clothing color patterns, movement patterns
- **Implementation**: Siamese network or embedding approach
- **Privacy Constraints**: No biometric data, no personal identification

**Expected Outputs:**
- Staff identifier (e.g., "Person A", "Person B")
- Confidence score for identification
- Consistent tracking across sessions

**Edge Cases to Handle:**
- Staff changing uniforms/clothes
- Multiple staff with similar appearance
- New staff members
- Changing lighting conditions

**Dependencies:**
- Feature extraction pipeline
- Clustering algorithm for identification
- Privacy-preserving design review

### 4. Transaction Correlation with Visual Events

**Technical Implementation:**
- **Data Integration**: API connection to POS system
- **Time Alignment**: NTP or manual time synchronization between systems
- **Correlation Logic**:
  - Match inventory access with transaction timestamps
  - Associate item removal with transaction line items
  - Calculate time deltas between visual events and transactions

**Expected Outputs:**
- Matched event-transaction pairs
- Unmatched events (potential issues)
- Correlation confidence scores
- Time gap metrics

**Edge Cases to Handle:**
- Transaction entered later than actual sale
- Batch transaction entry
- System time differences
- Transactions for pre-picked items

**Dependencies:**
- POS API integration
- Reliable timestamp synchronization
- Matching algorithm development

### 5. Pattern Recognition for Common Theft Behaviors

**Technical Implementation:**
- **Sequence Modeling**: LSTM or transformer architecture for temporal patterns
- **Pre-defined Patterns**:
  - Concealment motions (characteristic body postures)
  - Frequent access without transactions
  - Access to multiple areas in quick succession
  - After-hours or unusual timing patterns
- **Anomaly Detection**: Isolation Forest or One-Class SVM for unusual patterns

**Expected Outputs:**
- Pattern match alerts with pattern type
- Confidence scores
- Supporting evidence (event sequence)
- Anomaly scores for unusual behaviors

**Edge Cases to Handle:**
- Legitimate variations of suspicious patterns
- New theft methods not in training data
- Pattern variations across different staff

**Dependencies:**
- Sequence modeling framework
- Pattern definition library
- Anomaly detection pipeline

## Phase 3: Advanced Detection Capabilities

### 1. Advanced Behavioral Analysis

**Technical Implementation:**
- **Video Understanding**: SlowFast network or 3D CNN architecture
- **Activity Recognition**: 
  - Fine-grained action classification (20+ actions)
  - Long-range temporal modeling (1-5 minutes)
- **Contextual Understanding**:
  - Scene graph generation
  - Relationship detection between people and objects
- **Attention Mechanisms**: For focusing on relevant interactions

**Expected Outputs:**
- Complex activity classifications
- Multi-step action sequences
- Behavioral anomaly scores
- Contextual relationship mapping

**Edge Cases to Handle:**
- Novel behaviors
- Partial occlusion of key actions
- Compound or overlapping activities
- Intentional concealment

**Dependencies:**
- Advanced AI models (larger models)
- Potential hardware upgrades
- Extensive training data

### 2. Predictive Risk Assessment

**Technical Implementation:**
- **Machine Learning Approach**: Gradient boosting or neural network
- **Features**:
  - Historical patterns of staff behavior
  - Temporal features (time of day, day of week)
  - Inventory characteristics (value, size, popularity)
  - Transaction volume and patterns
- **Risk Scoring**: 0-100 scale with configurable thresholds
- **Early Warning System**: Detect precursors to theft events

**Expected Outputs:**
- Real-time risk scores
- Contributing factor breakdown
- Trend analysis over time
- Early warning alerts

**Edge Cases to Handle:**
- Seasonal variations in normal behavior
- Special events (inventory, sales)
- New staff learning patterns
- Legitimate but unusual activities

**Dependencies:**
- Historical data accumulation
- Feature engineering pipeline
- Model validation framework

### 3. Inventory Reconciliation Assistance

**Technical Implementation:**
- **Visual Counting**: Object detection with counting
- **Inventory Tracking**:
  - Track items entering/leaving the van
  - Track items sold via transactions
  - Calculate expected inventory levels
- **Discrepancy Detection**: Compare visual inventory with expected inventory
- **Confidence Intervals**: Account for detection uncertainty

**Expected Outputs:**
- Visual inventory estimates
- Discrepancy reports
- Confidence levels for estimates
- Flagged areas with highest variance

**Edge Cases to Handle:**
- Items moved but not visible to camera
- Similar-looking items
- Densely packed inventory
- Partial visibility of inventory areas

**Dependencies:**
- Object counting models
- Inventory system integration
- Statistical reconciliation methods

### 4. Multi-person Interaction Analysis

**Technical Implementation:**
- **Tracking**: Multi-object tracking with occlusion handling
- **Interaction Detection**:
  - Proximity analysis
  - Joint activity recognition
  - Object handoff detection
- **Role Classification**: Detect customer vs. staff vs. unauthorized person
- **Suspicious Interaction Patterns**: 
  - Unusual proximity or duration
  - Concealed hand movements between people
  - Unusual timing of interactions

**Expected Outputs:**
- Interaction event logs
- Role classifications
- Suspicious interaction alerts
- Relationship maps between individuals

**Edge Cases to Handle:**
- Crowded scenarios
- Brief interactions
- Partial visibility
- Normal variations in interaction style

**Dependencies:**
- Multi-person tracking system
- Interaction detection models
- Role classification algorithms

## Technical Implementation Considerations

### Model Selection Strategy
- Prefer lightweight models for edge deployment
- Consider model ensemble approaches for higher accuracy
- Evaluate TensorFlow Lite, ONNX Runtime, and TensorRT paths

### Testing and Validation
- Create synthetic test scenarios for rare events
- Implement performance metrics for each capability
- Establish accuracy thresholds for production readiness

### Training Data Requirements
- Phase 1: Can use public datasets with fine-tuning
- Phase 2: Requires 50-100 hours of van-specific footage
- Phase 3: Requires 200+ hours with annotated events

### Processing Requirements
- Phase 1: 2-3 FPS processing sufficient
- Phase 2: 5-10 FPS recommended
- Phase 3: Consider batched processing for complex models

### Privacy Considerations
- Implement face blurring for customers
- Store only metadata and event clips, not continuous footage
- Design all staff identification to be role-based, not identity-based
