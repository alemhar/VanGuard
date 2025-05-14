# SmartVan Monitor: Software Specifications

## System Architecture Overview

### Core Components
1. **Video Capture System**
   - Camera feed acquisition
   - Frame preprocessing (resizing, normalization)
   - Motion detection for recording triggers

2. **Detection Engine**
   - Computer vision models for object/action detection
   - Temporal analysis for behavior patterns
   - Event classification and prioritization

3. **Local Storage System**
   - Database for events and metadata
   - Video clip storage management
   - Storage optimization and cleanup

4. **Synchronization Module**
   - Connectivity management
   - Data transfer prioritization
   - Backend API integration

## Software Flow

### 1. Video Processing Pipeline
```
Raw Camera Feed → Motion Detection → Frame Preprocessing → Object Detection → 
Event Detection → Event Classification → Storage
```

#### Detailed Steps:
1. **Camera Feed Acquisition**
   - Continuous feed from both cameras at 10-15 FPS
   - Hardware acceleration for decoding (when available)
   - Auto-recovery for camera disconnection

2. **Motion Detection**
   - Background subtraction for motion detection
   - Dynamic sensitivity adjustment based on time of day
   - Motion-triggered recording to conserve storage

3. **Frame Preprocessing**
   - Resize to 640x480 for processing
   - Color correction and normalization
   - Frame stabilization (if needed)

4. **Object Detection**
   - Detect relevant objects (people, boxes, products)
   - Track objects across frames
   - Calculate object interaction zones

5. **Event Detection & Classification**
   - Classify detected activities (box opening, item removal)
   - Assign confidence scores to detections
   - Tag events with metadata (timestamp, duration, involved objects)

### 2. Event Processing Workflow
```
Detected Event → Business Rules Engine → Classification → 
Prioritization → Storage/Alert Generation
```

#### Business Rules Engine
- Rule 1: Access to inventory without corresponding POS activity within 30 minutes (potential theft)
- Rule 2: High-frequency access to same inventory area (>3 accesses within 15 minutes) without proportional sales activity (suspicious)
- Rule 3: After-hours inventory access (outside of 7AM-7PM or configured business hours) without manager override (unauthorized)
- Rule 4: Removal of items from inventory detected but no transaction recorded within 30 minutes (potential theft)
- Rule 5: Pattern matching against known theft behaviors (repeated small inventory access followed by breaks, accessing multiple inventory areas within 2 minutes, concealment gestures detected)

#### Event Classification
| Classification | Description | Priority | Storage Policy |
|----------------|-------------|----------|---------------|
| Critical | High confidence theft event | Highest | Full video + metadata |
| Suspicious | Potential theft or unauthorized access | High | 30-sec clips before/after + metadata |
| Unusual | Abnormal but not clearly suspicious | Medium | Short clip + metadata |
| Routine | Normal operation | Low | Metadata only |

### 3. Storage Management
```
Event Data → SQLite Database
Video Clips → Filesystem (with metadata references)
```

#### Database Schema (Enhanced)
- **Events Table**
  - EventID (Primary Key)
  - Timestamp
  - EventType
  - Confidence
  - Description
  - SyncStatus (PENDING, IN_PROGRESS, COMPLETED, FAILED, PARTIAL)
  - LastSyncAttempt (timestamp)
  - SyncRetryCount (integer)
  - Priority (integer)

- **Inventory Access Table**
  - AccessID (Primary Key)
  - EventID (Foreign Key)
  - InventoryLocation
  - Duration
  - ItemsDetected

- **Media Files Table**
  - MediaID (Primary Key)
  - EventID (Foreign Key)
  - FilePath
  - FileSize
  - MediaType (video, image)
  - Duration (for videos)
  - SyncStatus (separate from event sync)
  - SyncPriority
  - LastSyncAttempt (timestamp)

- **Sync Log Table**
  - SyncID (Primary Key)
  - Timestamp
  - ConnectionType (WiFi, 4G, etc.)
  - SyncStatus
  - DataSent (bytes)
  - EventsSynced (count)
  - MediaSynced (count)
  - Duration (seconds)
  - ErrorMessage (nullable)

#### Storage Policies
- Implement circular buffer for video storage
- Prioritize critical events for long-term storage
- Auto-delete routine events after sync
- Low space contingency policies

### 4. Synchronization Process
```
Connectivity Detection → Data Preparation → Priority-based Transfer →
Confirmation → Local Cleanup
```

#### Sync Triggers
- Connection to depot WiFi network
- Scheduled attempts via cellular (configurable)
- Manual trigger via admin interface
- Critical event detection (if online)

#### Priority Queue
1. System health data and logs
2. Critical event metadata
3. High-priority event video clips
4. Medium-priority event metadata
5. Medium-priority event video clips
6. Low-priority metadata

#### Bandwidth Management
- Adaptive transfer rates based on connection quality
- Compression of video before transfer
- Incremental sync from last successful point
- Resume capability for interrupted transfers

## Key Software Features

### 1. Detection Capabilities

#### Phase 1 (MVP)
- Basic motion detection in inventory areas
- Person presence detection
- Inventory access event detection
- Simple temporal patterns (access frequency)

#### Phase 2
- Box opening detection
- Item removal tracking
- Staff identification (anonymized)
- Transaction correlation with visual events
- Pattern recognition for common theft behaviors

#### Phase 3
- Advanced behavioral analysis
- Predictive risk assessment
- Inventory reconciliation assistance
- Multi-person interaction analysis

### 2. Offline Intelligence

#### Local Decision Making
- Event classification without backend connectivity
- Smart storage management based on event importance
- Adaptive recording policies based on battery and storage
- Self-healing capabilities for system issues

#### Edge Computing Features
- Local model execution on Raspberry Pi
- Dynamic model loading based on detection needs
- Model quantization for performance optimization
- Thermal management for continuous operation

### 3. User Interface Components

#### Admin Mobile App (Future)
- Remote system status checking
- Live view when in proximity
- Manual sync triggering
- Alert management and response

#### Web Dashboard (Backend Integration)
- Fleet-wide event visualization
- Theft pattern analysis
- ROI calculations
- Evidence management for investigations

### 4. Security & Privacy

#### Data Protection
- Encrypted local storage
- Secure transfer protocols
- Role-based access control
- Data retention policies

#### Privacy Considerations
- Focus on inventory rather than people
- Anonymization of staff where possible
- Configurable privacy zones
- Transparent monitoring policies

## Integration API

### Backend Communication Protocol
```
REST API with JWT authentication
```

### Key Endpoints
- `/api/sync/events` - Send event metadata
- `/api/sync/media` - Upload video evidence
- `/api/system/status` - Report system health
- `/api/config/update` - Receive configuration updates

### Data Formats
- Event data: JSON
- Video clips: MP4 (H.264)
- System logs: Structured JSON
- Configuration: JSON

## Development Tools & Libraries

### Core Vision Stack
- OpenCV for image processing
- TensorFlow Lite for model execution
- PyTorch (optional for more complex models)

### Backend & Storage
- SQLite for local database
- Flask for local API (if needed)
- Requests for HTTP communication

### System Management
- Systemd for service management
- Docker for component isolation (optional)
- Logging: Python logging + rotation

## Performance Optimization

### Resource Management
- Dynamic FPS adjustment based on activity
- Selective recording based on events
- Sleep modes during inactive periods
- Model switching based on detection needs

### Memory Management
- Limit concurrent processing tasks
- Efficient tensor operations
- Garbage collection optimization
- Memory monitoring and overflow prevention
