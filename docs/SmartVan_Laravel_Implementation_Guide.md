# SmartVan Anomaly Detection: Laravel Implementation Guide

## Overview

This guide outlines how to implement the SmartVan anomaly detection system using a Laravel backend with Vue.js and Inertia.js for the frontend. This stack provides a powerful foundation for real-time monitoring, data processing, and user interface components.

## Tech Stack Components

### Backend (Laravel)

- **Laravel 10+**: PHP framework for backend logic and API endpoints
- **Laravel Sanctum**: For API authentication
- **Laravel Queues**: For processing background jobs (event re-evaluation)
- **Laravel Horizon**: For queue monitoring and management
- **Laravel Scheduler**: For time-based anomaly resolution

### Frontend (Vue + Inertia)

- **Vue.js 3**: For reactive UI components
- **Inertia.js**: For SPA-like functionality without building a separate API
- **Tailwind CSS**: For responsive UI design
- **Chart.js**: For visualizing anomaly data and patterns
- **Polling**: For dashboard updates (since we're using webhooks, not WebSockets)

## Database Schema

### Key Tables

1. **events**
   ```php
   Schema::create('events', function (Blueprint $table) {
       $table->id();
       $table->string('event_id')->unique();
       $table->string('van_id');
       $table->string('camera_id');
       $table->timestamp('event_timestamp');
       $table->string('event_type');
       $table->json('human_detection');
       $table->json('inventory');
       $table->json('raw_data');
       $table->timestamps();
       
       $table->index(['van_id', 'event_timestamp']);
   });
   ```

2. **pos_transactions**
   ```php
   Schema::create('pos_transactions', function (Blueprint $table) {
       $table->id();
       $table->string('transaction_id')->unique();
       $table->string('van_id');
       $table->timestamp('transaction_timestamp');
       $table->decimal('amount', 10, 2);
       $table->json('items');
       $table->json('raw_data');
       $table->timestamps();
       
       $table->index(['van_id', 'transaction_timestamp']);
   });
   ```

3. **event_transaction_matches**
   ```php
   Schema::create('event_transaction_matches', function (Blueprint $table) {
       $table->id();
       $table->foreignId('event_id')->constrained();
       $table->foreignId('transaction_id')->constrained('pos_transactions');
       $table->timestamp('matched_at');
       $table->string('match_type')->default('AUTOMATIC'); // AUTOMATIC or MANUAL
       $table->string('matched_by')->nullable(); // User ID if manual
       $table->timestamps();
       
       $table->unique(['event_id', 'transaction_id']);
   });
   ```

4. **anomalies**
   ```php
   Schema::create('anomalies', function (Blueprint $table) {
       $table->id();
       $table->foreignId('event_id')->constrained();
       $table->decimal('anomaly_score', 4, 2);
       $table->string('anomaly_level'); // HIGH, MEDIUM, LOW
       $table->string('anomaly_status'); // PROVISIONAL, CONFIRMED, CLEARED
       $table->json('anomaly_reasons');
       $table->boolean('re_evaluation_needed')->default(false);
       $table->string('missing_data_source')->nullable();
       $table->timestamp('last_evaluated_at');
       $table->timestamp('finalized_at')->nullable();
       $table->timestamps();
       
       $table->index(['anomaly_status', 'missing_data_source']);
   });
   ```

## Implementation Components

### 1. API Endpoints

Create dedicated API endpoints for receiving data from SmartVan devices:

```php
// routes/api.php
Route::middleware('auth:sanctum')->prefix('v1')->group(function () {
    Route::post('/events', [EventController::class, 'store']);
    Route::post('/heartbeats', [HeartbeatController::class, 'store']);
    Route::post('/transactions', [TransactionController::class, 'store']);
});
```

### 2. Event Controller

```php
namespace App\Http\Controllers;

use App\Models\Event;
use App\Jobs\ProcessEventForAnomalies;
use Illuminate\Http\Request;

class EventController extends Controller
{
    public function store(Request $request)
    {
        $validated = $request->validate([
            'event_id' => 'required|string|unique:events,event_id',
            'van_id' => 'required|string',
            'camera' => 'required|string',
            'timestamp' => 'required|integer',
            'event_type' => 'required|string',
            // Additional validation rules
        ]);
        
        // Store the event
        $event = Event::create([
            'event_id' => $validated['event_id'],
            'van_id' => $validated['van_id'],
            'camera_id' => $validated['camera'],
            'event_timestamp' => $validated['timestamp'],
            'event_type' => $validated['event_type'],
            'human_detection' => json_encode($request->input('human_detection', [])),
            'inventory' => json_encode($request->input('inventory', [])),
            'raw_data' => json_encode($request->all())
        ]);
        
        // Dispatch a job to analyze this event
        ProcessEventForAnomalies::dispatch($event);
        
        return response()->json(['status' => 'success', 'message' => 'Event received']);
    }
}
```

### 3. Anomaly Detection Service

```php
namespace App\Services;

use App\Models\Event;
use App\Models\PosTransaction;
use App\Models\Anomaly;
use Carbon\Carbon;

class AnomalyDetectionService
{
    public function analyzeEvent(Event $event)
    {
        $anomalyScore = 0;
        $anomalyReasons = [];
        
        // Check for human detection with YOLO
        if ($this->hasHighConfidenceHumanDetection($event)) {
            $anomalyScore += 0.35 * $this->getHumanDetectionConfidence($event);
            $anomalyReasons[] = 'HIGH_CONFIDENCE_HUMAN_DETECTION';
        }
        
        // Check for after-hours activity
        if (!$this->isBusinessHours($event->event_timestamp)) {
            $anomalyScore += 0.25;
            $anomalyReasons[] = 'AFTER_HOURS_ACTIVITY';
        }
        
        // Check for high-frequency access
        $recentAccesses = $this->getRecentAccessCount($event);
        if ($recentAccesses > 3) {
            $anomalyScore += 0.20 * ($recentAccesses / 3);
            $anomalyReasons[] = 'HIGH_FREQUENCY_ACCESS';
        }
        
        // Check for inventory removal without POS transaction
        if ($this->isItemRemovalEvent($event)) {
            $transaction = $this->findUnmatchedPosTransaction($event);
            
            if ($transaction) {
                // Create match record to prevent reuse
                $this->createEventTransactionMatch($event, $transaction);
            } else {
                // No matching transaction found
                $anomalyScore += 0.20;
                $anomalyReasons[] = 'REMOVAL_WITHOUT_TRANSACTION';
                
                // Check if this might be due to missing POS data
                $needsReevaluation = $this->mightHaveMissingPosData($event);
                
                return $this->createAnomalyRecord(
                    $event,
                    $anomalyScore,
                    $anomalyReasons,
                    $needsReevaluation ? 'PROVISIONAL' : 'CONFIRMED',
                    $needsReevaluation ? 'POS_DATA' : null
                );
            }
        }
        
        // Only create anomaly record if score is significant
        if ($anomalyScore >= 0.2) {
            return $this->createAnomalyRecord(
                $event, 
                $anomalyScore, 
                $anomalyReasons, 
                'CONFIRMED',
                null
            );
        }
        
        return null;
    }
    
    protected function findUnmatchedPosTransaction(Event $event)
    {
        $windowStart = Carbon::createFromTimestamp($event->event_timestamp)->subMinutes(30);
        $windowEnd = Carbon::createFromTimestamp($event->event_timestamp)->addMinutes(30);
        
        // Find transactions in the window that haven't been matched yet
        return PosTransaction::where('van_id', $event->van_id)
            ->whereBetween('transaction_timestamp', [$windowStart, $windowEnd])
            ->whereDoesntHave('eventMatches') // Only get unmatched transactions
            ->first();
    }
    
    protected function createEventTransactionMatch(Event $event, PosTransaction $transaction)
    {
        return \App\Models\EventTransactionMatch::create([
            'event_id' => $event->id,
            'transaction_id' => $transaction->id,
            'matched_at' => now(),
            'match_type' => 'AUTOMATIC'
        ]);
    }
    
    protected function createAnomalyRecord(Event $event, $score, $reasons, $status, $missingDataSource = null)
    {
        return Anomaly::create([
            'event_id' => $event->id,
            'anomaly_score' => $score,
            'anomaly_level' => $this->getAnomalyLevel($score),
            'anomaly_status' => $status,
            'anomaly_reasons' => json_encode($reasons),
            're_evaluation_needed' => $status === 'PROVISIONAL',
            'missing_data_source' => $missingDataSource,
            'last_evaluated_at' => now()
        ]);
    }
    
    // Additional helper methods...
}
```

### 4. Re-evaluation Job

```php
namespace App\Jobs;

use App\Models\Anomaly;
use App\Services\AnomalyDetectionService;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;

class ReEvaluateProvisionalAnomalies implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    protected $dataSource;
    protected $dataId;

    public function __construct($dataSource, $dataId)
    {
        $this->dataSource = $dataSource;
        $this->dataId = $dataId;
    }

    public function handle(AnomalyDetectionService $anomalyService)
    {
        // Find provisional anomalies needing this data source
        $anomalies = Anomaly::where('anomaly_status', 'PROVISIONAL')
            ->where('missing_data_source', $this->dataSource)
            ->where('re_evaluation_needed', true)
            ->get();
            
        foreach ($anomalies as $anomaly) {
            // Re-analyze the event with fresh data
            $event = $anomaly->event;
            $newAnalysis = $anomalyService->analyzeEvent($event);
            
            // Update the existing anomaly with new results
            $anomaly->update([
                'anomaly_score' => $newAnalysis->anomaly_score,
                'anomaly_level' => $newAnalysis->anomaly_level,
                'anomaly_status' => 'FINAL',
                'anomaly_reasons' => $newAnalysis->anomaly_reasons,
                're_evaluation_needed' => false,
                'missing_data_source' => null,
                'last_evaluated_at' => now(),
                'finalized_at' => now()
            ]);
            
            // If the status changed, notify relevant users
            if ($anomaly->anomaly_level != $newAnalysis->anomaly_level) {
                // Dispatch notification
            }
        }
    }
}
```

### 5. Laravel Command to Finalize Old Anomalies

```php
namespace App\Console\Commands;

use App\Models\Anomaly;
use Illuminate\Console\Command;
use Carbon\Carbon;

class FinalizeOldAnomalies extends Command
{
    protected $signature = 'anomalies:finalize';
    protected $description = 'Finalize provisional anomalies older than 24 hours';

    public function handle()
    {
        $cutoff = Carbon::now()->subDay();
        
        $count = Anomaly::where('anomaly_status', 'PROVISIONAL')
            ->where('created_at', '<', $cutoff)
            ->update([
                'anomaly_status' => 'CONFIRMED',
                're_evaluation_needed' => false,
                'finalized_at' => Carbon::now()
            ]);
            
        $this->info("Finalized {$count} old provisional anomalies");
        
        return Command::SUCCESS;
    }
}
```

### 6. Vue Components for Anomaly Dashboard

#### AnomalyDashboard.vue

```vue
<template>
  <app-layout>
    <div class="py-12">
      <div class="max-w-7xl mx-auto sm:px-6 lg:px-8">
        <div class="bg-white overflow-hidden shadow-xl sm:rounded-lg p-6">
          <h1 class="text-2xl font-bold mb-6">SmartVan Anomaly Dashboard</h1>
          
          <!-- Stats Cards -->
          <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            <stats-card 
              title="High Priority" 
              :count="stats.highPriority" 
              color="red" 
              icon="exclamation-circle"
            />
            <stats-card 
              title="Medium Priority" 
              :count="stats.mediumPriority" 
              color="yellow" 
              icon="exclamation"
            />
            <stats-card 
              title="Provisional" 
              :count="stats.provisional" 
              color="blue" 
              icon="clock"
            />
            <stats-card 
              title="Total Events" 
              :count="stats.totalEvents" 
              color="gray" 
              icon="document-text"
            />
          </div>
          
          <!-- Filter Controls -->
          <div class="mb-6 flex flex-wrap gap-2">
            <filter-select 
              v-model="filters.status" 
              :options="statusOptions" 
              label="Status"
            />
            <filter-select 
              v-model="filters.van" 
              :options="vanOptions" 
              label="Van"
            />
            <date-range-picker v-model="filters.dateRange" />
            <button 
              class="bg-indigo-600 text-white px-4 py-2 rounded-md hover:bg-indigo-700"
              @click="applyFilters"
            >
              Apply Filters
            </button>
          </div>
          
          <!-- Anomalies Table -->
          <anomalies-table 
            :anomalies="anomalies" 
            :loading="loading"
            @view-details="viewAnomalyDetails"
          />
          
          <!-- Pagination -->
          <pagination :links="links" class="mt-6" />
        </div>
      </div>
    </div>
    
    <!-- Anomaly Details Modal -->
    <anomaly-details-modal 
      v-if="selectedAnomaly" 
      :anomaly="selectedAnomaly"
      @close="selectedAnomaly = null"
      @resolve="resolveAnomaly"
    />
  </app-layout>
</template>

<script>
export default {
  data() {
    return {
      anomalies: [],
      stats: {
        highPriority: 0,
        mediumPriority: 0,
        provisional: 0,
        totalEvents: 0
      },
      filters: {
        status: 'all',
        van: 'all',
        dateRange: {
          start: null,
          end: null
        }
      },
      loading: false,
      links: {},
      selectedAnomaly: null,
      statusOptions: [
        { value: 'all', label: 'All Statuses' },
        { value: 'PROVISIONAL', label: 'Provisional' },
        { value: 'CONFIRMED', label: 'Confirmed' },
        { value: 'CLEARED', label: 'Cleared' }
      ],
      vanOptions: [
        { value: 'all', label: 'All Vans' }
        // Populated from props or API
      ]
    };
  },
  
  mounted() {
    this.fetchAnomalies();
    this.fetchStats();
    
    // Set up Echo for real-time updates
    // Set up polling for dashboard updates
    this.startPolling();
  },
  
  methods: {
    fetchAnomalies() {
      this.loading = true;
      axios.get(route('api.anomalies.index', this.getQueryParams()))
        .then(response => {
          this.anomalies = response.data.data;
          this.links = response.data.links;
        })
        .catch(error => {
          console.error('Error fetching anomalies:', error);
        })
        .finally(() => {
          this.loading = false;
        });
    },
    
    fetchStats() {
      axios.get(route('api.anomalies.stats'))
        .then(response => {
          this.stats = response.data;
        });
    },
    
    applyFilters() {
      this.fetchAnomalies();
    },
    
    getQueryParams() {
      const params = {
        page: 1
      };
      
      if (this.filters.status !== 'all') {
        params.status = this.filters.status;
      }
      
      if (this.filters.van !== 'all') {
        params.van_id = this.filters.van;
      }
      
      if (this.filters.dateRange.start) {
        params.start_date = this.filters.dateRange.start;
      }
      
      if (this.filters.dateRange.end) {
        params.end_date = this.filters.dateRange.end;
      }
      
      return params;
    },
    
    viewAnomalyDetails(anomaly) {
      this.selectedAnomaly = anomaly;
    },
    
    resolveAnomaly(anomalyId, resolution) {
      axios.post(route('api.anomalies.resolve', anomalyId), { resolution })
        .then(() => {
          this.selectedAnomaly = null;
          this.fetchAnomalies();
          this.fetchStats();
        });
    },
    
    startPolling() {
      // Poll for updates every 30 seconds
      this.pollingInterval = setInterval(() => {
        this.fetchAnomalies();
        this.fetchStats();
      }, 30000);
    },
    
    beforeUnmount() {
      // Clear polling interval when component is destroyed
      if (this.pollingInterval) {
        clearInterval(this.pollingInterval);
      }
    },
    
    showNotification(title, message) {
      if ('Notification' in window && Notification.permission === 'granted') {
        new Notification(title, {
          body: message,
          icon: '/notification-icon.png'
        });
      }
    }
  }
};
</script>
```

### 7. Scheduled Tasks Setup

```php
// app/Console/Kernel.php
protected function schedule(Schedule $schedule)
{
    $schedule->command('anomalies:finalize')
             ->hourly();
             
    // Cleanup old events and logs
    $schedule->command('events:cleanup --older-than=90')
             ->daily();
}
```

### 8. Webhook Implementation

The SmartVan system communicates with the backend using webhooks. Here's how to implement the webhook endpoints and their processing logic:

#### 8.1 Webhook Routes Configuration

```php
// routes/api.php
Route::middleware('auth:sanctum')->prefix('v1')->group(function () {
    // Webhook endpoints for SmartVan data
    Route::post('/events', [EventController::class, 'store']);
    Route::post('/heartbeats', [HeartbeatController::class, 'store']);
    Route::post('/metrics', [MetricsController::class, 'store']);
    
    // Transaction endpoints for POS system
    Route::post('/transactions', [TransactionController::class, 'store']);
    
    // Feedback endpoint to acknowledge receipt
    Route::post('/acknowledge/{event_id}', [EventController::class, 'acknowledge']);
});
```

#### 8.2 Webhook Processing Strategy

```php
namespace App\Http\Controllers;

use App\Models\Event;
use App\Jobs\ProcessEventForAnomalies;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Log;

class EventController extends Controller
{
    public function store(Request $request)
    {
        // Log incoming webhook data for debugging
        Log::debug('Received event webhook', ['data' => $request->all()]);
        
        // Validate the webhook data
        $validated = $request->validate([
            'event_id' => 'required|string',
            'van_id' => 'required|string',
            'camera' => 'required|string',
            'timestamp' => 'required|integer',
            'event_type' => 'required|string',
            'human_detection' => 'required|array',
            'human_detection.detected' => 'required|boolean',
            'human_detection.confidence' => 'required|numeric|min:0|max:1',
            'human_detection.detection_method' => 'required|string'
        ]);
        
        // Handle duplicate events gracefully (important for webhook reliability)
        $existingEvent = Event::where('event_id', $validated['event_id'])->first();
        if ($existingEvent) {
            // Return success to avoid the sender retrying
            return response()->json([
                'status' => 'success', 
                'message' => 'Event already processed',
                'duplicate' => true
            ]);
        }
        
        // Extract YOLO detection data
        $human_detection = $request->input('human_detection', []);
        $yolo_detection = $human_detection['detection_method'] === 'YOLO';
        
        // Process vibration data using your optimized thresholds
        $is_vibration = false;
        $motion_data = $request->input('motion', []);
        if (isset($motion_data['intensity']) && $motion_data['intensity'] < 40) {
            $is_vibration = true;
        }
        
        // Store the event
        $event = Event::create([
            'event_id' => $validated['event_id'],
            'van_id' => $validated['van_id'],
            'camera_id' => $validated['camera'],
            'event_timestamp' => $validated['timestamp'],
            'event_type' => $validated['event_type'],
            'human_detection' => json_encode($human_detection),
            'inventory' => json_encode($request->input('inventory', [])),
            'is_vibration' => $is_vibration,
            'has_yolo_detection' => $yolo_detection,
            'raw_data' => json_encode($request->all())
        ]);
        
        // Dispatch a job to analyze this event
        // Use a queue to avoid webhook timeout
        ProcessEventForAnomalies::dispatch($event);
        
        // Immediately return a response to the webhook sender
        return response()->json([
            'status' => 'success', 
            'message' => 'Event received and queued for processing',
            'event_id' => $validated['event_id']
        ]);
    }
    
    public function acknowledge(Request $request, $event_id)
    {
        // Update the event to mark it as acknowledged
        $event = Event::where('event_id', $event_id)->first();
        
        if ($event) {
            $event->update(['acknowledged_at' => now()]);
            return response()->json(['status' => 'success']);
        }
        
        return response()->json(['status' => 'error', 'message' => 'Event not found'], 404);
    }
}
```

## Deployment Considerations

### Docker Setup

```yaml
# docker-compose.yml
version: '3'
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    image: smartvan-app
    container_name: smartvan-app
    restart: unless-stopped
    volumes:
      - .:/var/www/html
    networks:
      - smartvan

  queue:
    image: smartvan-app
    container_name: smartvan-queue
    restart: unless-stopped
    command: php artisan queue:work --tries=3
    volumes:
      - .:/var/www/html
    networks:
      - smartvan

  # No websockets container needed for webhook-based communication

  scheduler:
    image: smartvan-app
    container_name: smartvan-scheduler
    restart: unless-stopped
    command: php artisan schedule:work
    volumes:
      - .:/var/www/html
    networks:
      - smartvan

  # Other services: nginx, mysql, redis, etc.
```

## Performance Optimization

1. **Database Indexes**
   - Ensure all foreign keys are indexed
   - Create composite indexes for frequently queried combinations

2. **Caching Strategy**
   - Cache van list, transaction summaries, and dashboard statistics
   - Use Redis for high-speed caching

3. **Queue Configuration**
   - Use Redis queues for better performance
   - Set appropriate retry and timeout settings

4. **Database Connection Pooling**
   - Configure your database to handle multiple connections efficiently

## Security Considerations

1. **API Authentication**
   - Use Laravel Sanctum for token-based authentication
   - Implement API rate limiting

2. **CORS Configuration**
   - Properly configure Cross-Origin Resource Sharing

3. **Encryption**
   - Ensure all sensitive data is encrypted at rest
   - Use HTTPS for all communications

4. **User Roles and Permissions**
   - Implement granular access control
   - Log all security-relevant actions

## Monitoring and Logging

1. **Implement Laravel Telescope** for development debugging
2. **Use Laravel Horizon** for queue monitoring
3. **Set up Slack notifications** for critical anomalies
4. **Configure structured logging** to a centralized log management system

## Conclusion

This implementation guide provides a foundation for building the SmartVan anomaly detection system using the Laravel/Vue/Inertia stack. The design focuses on:

1. **Reliability**: Through queues, retry logic, and time-based resolution
2. **Performance**: Using proper indexing, caching, and asynchronous processing 
3. **Security**: Implementing proper authentication and data protection
4. **Real-time capabilities**: With WebSockets and event broadcasting

The solution handles the specific challenges of the SmartVan system, particularly the need to correlate data from multiple sources and manage asynchronous data arrival while maintaining high accuracy in anomaly detection.
