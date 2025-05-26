# SmartVan Monitor: Anomaly Detection Guide

## Overview

This guide outlines how to utilize the data transmitted by the SmartVan Monitor system to identify potential anomalies, security incidents, and operational issues. By leveraging the enhanced event data and YOLO detection capabilities, backend systems can implement sophisticated anomaly detection algorithms to distinguish between normal operations and suspicious activities.

## Anomaly Categories

The SmartVan Monitor system can detect several categories of anomalies:

1. **Inventory Theft Events**
2. **Unauthorized Access**
3. **System Tampering**
4. **Operational Anomalies**
5. **Vehicle Misuse**

## Data Points for Anomaly Detection

### Primary Event Signals

| Data Point | JSON Path | Anomaly Indicator |
|------------|-----------|-------------------|
| Human detection without inventory access | `human_detection.detected = true` without `inventory.access_detected = true` | Potential unauthorized presence |
| Inventory access without human detection | `inventory.access_detected = true` without `human_detection.detected = true` | Potential system bypass or sensor failure |
| After-hours activity | Compare `timestamp` with business hours | Unauthorized access |
| High-frequency inventory access | Multiple events within short timeframe | Potential theft pattern |
| No POS transaction following removal | `inventory.change_type = "ITEM_REMOVED"` without POS event | Potential theft |

### Enhanced Detection with YOLO

The YOLO integration provides critical data for distinguishing genuine anomalies from false positives:

| YOLO Data | Purpose | Anomaly Detection Use |
|-----------|---------|----------------------|
| `human_detection.detection_method = "YOLO"` | Confirms visual human presence | Reduces false positives from vibration |
| `human_detection.confidence` | Confidence of human detection | Higher confidence = stronger anomaly signal |
| `yolo_metrics.objects_detected` | Objects detected in scene | Can identify suspicious items (e.g., unauthorized tools) |

## Anomaly Detection Rules

### Rule 1: Inventory Theft Detection

```json
IF event.event_type == "INVENTORY_CHANGE" 
   AND event.inventory.change_type == "ITEM_REMOVED" 
   AND event.human_detection.detected == true
   AND NO UNMATCHED POS_Transaction WITHIN 30 minutes BEFORE OR AFTER event.timestamp
THEN Flag as "POTENTIAL_THEFT"
```

> **Note**: A strict one-to-one relationship must be maintained between inventory events and POS transactions:
> 
> 1. Each POS transaction should only be matched to a single inventory event. Once a transaction has been matched, it should be marked as 'used' and not available for validating other events.
> 
> 2. Similarly, each inventory event should only be matched to a single POS transaction. Once an event has been validated by a transaction, it should not be counted as validated by any other transactions.
> 
> This bidirectional constraint prevents both scenarios: (a) using one transaction to justify multiple removals, and (b) using multiple transactions to obscure a single suspicious removal event.

### Rule 2: Unauthorized Access Detection

```json
IF event.timestamp IS OUTSIDE BusinessHours
   AND event.human_detection.detected == true
   AND event.human_detection.confidence > 0.7
   AND event.human_detection.detection_method == "YOLO"
THEN Flag as "UNAUTHORIZED_ACCESS"
```

### Rule 3: High-Frequency Access Pattern

```json
IF COUNT(events) > 3 FOR SAME van_id AND SAME inventory.zone_id
   WITHIN 15 minute window
   AND SUM(events WHERE inventory.change_type == "ITEM_REMOVED") > 0
THEN Flag as "SUSPICIOUS_ACCESS_PATTERN"
```

### Rule 4: System Tampering Detection

```json
IF heartbeat.system_hash != previous_system_hash
   OR heartbeat.detectors_status.detectors_active == false
THEN Flag as "POTENTIAL_SYSTEM_TAMPERING"
```

### Rule 5: Vibration vs. Human Detection

```json
IF event.event_type == "INVENTORY_CHANGE"
   AND event.human_detection.detected == false
   AND performance_metrics.false_positive_reasons.vibration > 0
THEN Flag as "LIKELY_VIBRATION_FALSE_POSITIVE"
```

## Implementation Guidance

### Baseline Establishment

Before detecting anomalies, establish normal behavior patterns:

1. Collect at least 2 weeks of operational data for each van
2. Analyze typical inventory access patterns during business hours
3. Establish baseline metrics for:
   - Average inventory accesses per day
   - Normal access duration
   - Typical item removal patterns
   - Business hours activity levels

### Weighted Scoring System

Implement a weighted scoring system for anomaly detection:

| Factor | Weight | Notes |
|--------|--------|-------|
| Human detection confidence | 0.35 | Higher confidence increases anomaly score |
| After-hours timing | 0.25 | Outside business hours increases score |
| Frequency of access | 0.20 | Higher frequency increases score |
| Detection method | 0.10 | YOLO detection more reliable than motion-only |
| Historical pattern match | 0.10 | Matching known theft patterns increases score |

Calculate an anomaly score:
```
anomaly_score = Σ(factor_weight × factor_value)
```

Flag events when:
- `anomaly_score > 0.7` = High-priority alert
- `0.4 ≤ anomaly_score ≤ 0.7` = Medium-priority alert
- `anomaly_score < 0.4` = Low-priority or no alert

### Minimizing False Positives

The SmartVan Monitor system is optimized to reduce false positives from vehicle vibrations:

1. **YOLO Human Confirmation**: 
   - Prioritize events with `human_detection.detection_method = "YOLO"`
   - These events have visual confirmation of human presence

2. **Motion Stability**:
   - The system requires 7 consecutive frames for motion confirmation
   - Helps filter out transient vibrations

3. **Intensity Thresholds**:
   - Low intensity vibrations (< 40) are automatically ignored
   - Events must exceed `significant_motion_threshold: 6.0`

4. **Combined Evidence**:
   - Strongest anomaly signals have both motion and YOLO detection
   - Check for `human_detection.detection_method = "COMBINED"`

## Real-Time vs. Batch Processing

### Real-Time Anomaly Detection

Implement these checks for immediate alerts:

1. After-hours access
2. High-confidence human detection without authorized access
3. System tampering indicators

### Batch Processing Anomalies

These patterns require historical data analysis:

1. Frequency analysis (multiple accesses within time window)
2. Correlation with POS data
3. Pattern matching against known theft behaviors

## Backend Implementation Example

```python
def find_unmatched_pos_transaction(event, pos_data):
    """
    Find an unmatched POS transaction within the time window of the event.
    Enforces the one-to-one relationship between events and transactions.
    
    Args:
        event (dict): The inventory event to find a matching transaction for
        pos_data (dict): Contains all POS transactions and matching status
        
    Returns:
        dict or None: Returns the matching transaction or None if no match found
    """
    event_timestamp = event.get('timestamp', 0)
    van_id = event.get('van_id')
    # Time window: 30 minutes before and after
    window_start = event_timestamp - (30 * 60)  # 30 minutes before
    window_end = event_timestamp + (30 * 60)    # 30 minutes after
    
    # Filter for transactions in the right time window and from the same van
    candidate_transactions = [
        t for t in pos_data.get('transactions', [])
        if window_start <= t.get('timestamp', 0) <= window_end
        and t.get('van_id') == van_id
        and not t.get('matched_to_event')  # Only consider unmatched transactions
        and not is_event_already_validated(event['event_id'])  # Ensure event isn't already validated
    ]
    
    # Return the first matching transaction if any found
    return candidate_transactions[0] if candidate_transactions else None


def mark_transaction_as_matched(transaction_id, event_id):
    """
    Mark a POS transaction as matched to prevent reuse for other events.
    
    Args:
        transaction_id (str): The ID of the POS transaction
        event_id (str): The ID of the event it was matched to
    """
    # In a real implementation, this would update a database
    # For example:
    # db.transactions.update_one(
    #     {'id': transaction_id},
    #     {'$set': {'matched_to_event': event_id, 'match_timestamp': time.time()}}
    # )
    print(f"Transaction {transaction_id} matched to event {event_id}")


def mark_event_as_validated(event_id, transaction_id):
    """
    Mark an inventory event as validated by a transaction to prevent validation by other transactions.
    
    Args:
        event_id (str): The ID of the event
        transaction_id (str): The ID of the validating transaction
    """
    # In a real implementation, this would update a database
    # For example:
    # db.events.update_one(
    #     {'event_id': event_id},
    #     {'$set': {'validated_by_transaction': transaction_id, 'validation_timestamp': time.time()}}
    # )
    print(f"Event {event_id} validated by transaction {transaction_id}")


def is_event_already_validated(event_id):
    """
    Check if an event has already been validated by a transaction.
    
    Args:
        event_id (str): The ID of the event to check
        
    Returns:
        bool: True if the event has already been validated
    """
    # In a real implementation, this would query a database
    # For example:
    # event = db.events.find_one({'event_id': event_id})
    # return event is not None and 'validated_by_transaction' in event
    return False  # Simplified for example


def analyze_event(event, van_history, pos_data):
    """Analyze an event for potential anomalies"""
    anomaly_score = 0
    anomaly_reasons = []
    
    # Check for human detection confidence
    if event.get('human_detection', {}).get('detected', False):
        confidence = event.get('human_detection', {}).get('confidence', 0)
        # Higher confidence increases anomaly score
        if confidence > 0.8:
            anomaly_score += 0.35 * (confidence)
            if event.get('human_detection', {}).get('detection_method') == 'YOLO':
                anomaly_reasons.append('HIGH_CONFIDENCE_HUMAN_DETECTION')
    
    # Check for after-hours activity
    timestamp = event.get('timestamp', 0)
    if not is_business_hours(timestamp):
        anomaly_score += 0.25
        anomaly_reasons.append('AFTER_HOURS_ACTIVITY')
    
    # Check for high-frequency access
    recent_accesses = get_recent_access_count(event, van_history)
    if recent_accesses > 3:
        anomaly_score += 0.20 * (recent_accesses / 3)
        anomaly_reasons.append('HIGH_FREQUENCY_ACCESS')
    
    # Check for inventory removal without POS transaction
    if (event.get('inventory', {}).get('change_type') == 'ITEM_REMOVED'):
        transaction = find_unmatched_pos_transaction(event, pos_data)
        if transaction:
            # Mark this transaction as used to prevent re-matching
            mark_transaction_as_matched(transaction['id'], event['event_id'])
            # Mark this event as validated by this transaction
            mark_event_as_validated(event['event_id'], transaction['id'])
        else:
            # No unmatched transaction found within time window
            anomaly_score += 0.20
            anomaly_reasons.append('REMOVAL_WITHOUT_TRANSACTION')
    
    return {
        'event_id': event.get('event_id'),
        'anomaly_score': anomaly_score,
        'anomaly_level': get_anomaly_level(anomaly_score),
        'anomaly_reasons': anomaly_reasons
    }
```

## Integration with Business Rules

Combine anomaly detection with business rules for enhanced security:

1. **POS Integration**:
   - Compare inventory removals with POS transactions
   - Alert on removals without corresponding sales

2. **Employee Schedule**:
   - Cross-reference access events with employee schedules
   - Flag access outside scheduled hours

3. **Location Awareness**:
   - Use vehicle GPS data to correlate with inventory access
   - Flag accesses in unauthorized or unusual locations

4. **Multi-camera Correlation**:
   - Look for sequential access patterns across multiple cameras
   - Flag suspicious movement patterns inside the vehicle

## Data Synchronization and Re-evaluation

The van and POS systems may experience connectivity issues or timing delays, leading to temporary data inconsistencies. To handle this situation, the anomaly detection system should implement a re-evaluation mechanism:

### Provisional Anomaly Classification

```python
def classify_provisional_anomaly(event):
    """Mark an anomaly as provisional if it's missing potential matching data"""
    # Check if we're within a recent timeframe (last 24 hours)
    event_time = datetime.fromtimestamp(event.get('timestamp', 0))
    current_time = datetime.now()
    time_diff = current_time - event_time
    
    if time_diff.total_seconds() < 86400:  # Within 24 hours
        # Mark as provisional - needs re-evaluation when more data arrives
        return {
            'event_id': event.get('event_id'),
            'anomaly_status': 'PROVISIONAL',
            'last_evaluated': current_time.isoformat(),
            're_evaluation_needed': True,
            'missing_data_source': 'POS_DATA'  # or 'VAN_DATA' depending on what's missing
        }
    else:
        # After 24 hours, finalize the anomaly classification
        return {
            'event_id': event.get('event_id'),
            'anomaly_status': 'CONFIRMED',
            'last_evaluated': current_time.isoformat(),
            're_evaluation_needed': False
        }
```

### Re-evaluation Process

When new data arrives from either the van or POS system, previously provisional anomalies should be re-evaluated:

```python
def re_evaluate_provisional_anomalies(new_data_source, new_data):
    """Re-evaluate provisional anomalies when new data arrives"""
    # Find all provisional anomalies that need this data source
    provisional_anomalies = get_provisional_anomalies(missing_data_source=new_data_source)
    
    results = []
    for anomaly in provisional_anomalies:
        # Get the original event
        original_event = get_event_by_id(anomaly['event_id'])
        
        # Re-run the full analysis with the new data
        if new_data_source == 'POS_DATA':
            updated_result = analyze_event(original_event, get_van_history(original_event), new_data)
        else:  # VAN_DATA
            updated_result = analyze_event(original_event, new_data, get_pos_data())
        
        # Update the anomaly status
        updated_result['re_evaluation_needed'] = False
        updated_result['last_evaluated'] = datetime.now().isoformat()
        updated_result['anomaly_status'] = 'FINAL'
        updated_result['previous_status'] = anomaly['anomaly_status']
        
        # Save the updated result
        save_updated_anomaly(updated_result)
        results.append(updated_result)
    
    return results
```

### Implementation Guidelines

1. **Data Processing Pipeline**:
   - Process all incoming data immediately upon receipt
   - Tag anomalies as "provisional" when detected with incomplete data
   - Set a re-evaluation flag with the missing data source type

2. **Data Arrival Triggers**:
   - When POS data arrives: Re-evaluate all provisional anomalies tagged as missing POS data
   - When van data arrives: Re-evaluate all provisional anomalies tagged as missing van data

3. **Time-Based Resolution**:
   - Set a final resolution time (e.g., 24 hours) after which provisional anomalies
     are automatically finalized even without matching data
   - This prevents an ever-growing backlog of provisional anomalies

4. **Notification Strategy**:
   - For provisional anomalies: Send low-priority notifications or hold until resolved
   - For confirmed anomalies: Send immediate high-priority alerts
   - When provisional anomalies are resolved (either confirmed or cleared): Update previous notifications

5. **Correlation IDs**:
   - Use correlation IDs across van events and POS transactions
   - These IDs help in quickly finding related data during re-evaluation

This approach ensures that temporary connectivity issues don't result in false anomaly alerts while still maintaining security by eventually finalizing anomaly decisions after a reasonable waiting period.

## Advanced Anomaly Detection Techniques

### Machine Learning Approaches

1. **Unsupervised Learning**:
   - Use clustering algorithms to identify unusual event patterns
   - Implement isolation forests to detect outliers in multi-dimensional data

2. **Supervised Learning**:
   - Train models on labeled historical incidents
   - Use classification algorithms to identify potential theft patterns

3. **Temporal Pattern Mining**:
   - Analyze sequences of events for suspicious patterns
   - Detect unusual timing or order of inventory access

### Fleet-Wide Analysis

For organizations with multiple vehicles:

1. **Cross-vehicle Comparison**:
   - Compare metrics across similar vehicles/routes
   - Flag vehicles with significantly different patterns

2. **Driver Behavior Profiling**:
   - Establish baseline behavior for each driver
   - Detect deviations from established patterns

## Conclusion

By leveraging the enhanced data from the SmartVan Monitor system, especially the YOLO detection capabilities, backend systems can effectively distinguish between normal operations and potential security incidents. The combination of real-time alerting and historical pattern analysis provides a robust framework for protecting vehicle inventory and ensuring operational security.

The most effective anomaly detection will combine multiple signals, with special emphasis on events that include both motion detection and YOLO-confirmed human presence, which significantly reduces false positives from vehicle vibrations.
