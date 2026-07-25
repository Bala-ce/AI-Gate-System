/**
 * AI Automated Vehicle Gate Registration System Dashboard
 * Main Script Logic
 */

// Helper: build a HH:MM:SS:MS AM/PM string from a Date object
function buildTimestamp(d) {
    const hh  = String(d.getHours()).padStart(2, '0');
    const mm  = String(d.getMinutes()).padStart(2, '0');
    const ss  = String(d.getSeconds()).padStart(2, '0');
    const ms  = String(d.getMilliseconds()).padStart(3, '0');
    const ampm = d.getHours() >= 12 ? 'PM' : 'AM';
    let h12 = d.getHours() % 12 || 12;
    return `${String(h12).padStart(2,'0')}:${mm}:${ss}:${ms} ${ampm}`;
}

// Function to update header and camera overlay clocks in real-time
function updateDynamicTime() {
    const now = new Date();
    
    // Header datetime: e.g. "Thu, Oct 12, 2024 - 10:45:30 AM"
    const dateOptions = { weekday: 'short', year: 'numeric', month: 'short', day: 'numeric' };
    const timeOptions = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true };
    
    const timeString = `${now.toLocaleDateString('en-US', dateOptions)} - ${now.toLocaleTimeString('en-US', timeOptions)}`;
    document.getElementById('current-datetime').textContent = timeString;
        
    // Camera feed timestamp: YYYY-MM-DD HH:MM:SS:MS
    const camYear = now.getFullYear();
    const camMon = String(now.getMonth() + 1).padStart(2, '0');
    const camDay = String(now.getDate()).padStart(2, '0');
    const camHr = String(now.getHours()).padStart(2, '0');
    const camMin = String(now.getMinutes()).padStart(2, '0');
    const camSec = String(now.getSeconds()).padStart(2, '0');
    const camMs  = String(now.getMilliseconds()).padStart(3, '0');
    const tsText = `${camYear}-${camMon}-${camDay} ${camHr}:${camMin}:${camSec}:${camMs}`;
    
    document.getElementById('camera-timestamp').textContent = tsText;
    const entryTs = document.getElementById('camera-timestamp-entry');
    if (entryTs) entryTs.textContent = tsText;
}

// Start time immediately and refresh every 100ms for millisecond display
updateDynamicTime();
setInterval(updateDynamicTime, 100);

/**
 * Placeholder function requested by the user.
 * Connected to Python OpenCV video stream endpoint.
 * @param {string} streamUrl - URL to the video source
 */
function updateCameraFeed(streamUrl) {
    console.log(`[System]: Connecting to camera stream feed: ${streamUrl}`);
    const videoElem = document.getElementById('security-camera-feed');
    const placeholder = document.getElementById('camera-placeholder');
    
    if (streamUrl) {
        // Switch views
        placeholder.classList.add('hidden');
        videoElem.classList.remove('hidden');
        videoElem.src = streamUrl;
    } else {
        // Revert back to placeholder UI
        placeholder.classList.remove('hidden');
        videoElem.classList.add('hidden');
        videoElem.src = '';
    }
}

// Expose strictly to window element so external scripts/consoles can reach it easily
window.updateCameraFeed = updateCameraFeed;


// Live Table Management Logic
const tableBody = document.getElementById('registration-table');

/**
 * Utility: Convert any time string to HH:MM:SS:MS AM/PM format.
 * Handles:
 *   "07:44:18:253 PM"  (new full format)  → "07:44:18:253 PM"
 *   "07:44 PM"         (old short format)  → "07:44:00:000 PM"
 *   "19:44"            (24-hr no AM/PM)    → "07:44:00:000 PM"
 *   "07:44:18 PM"      (seconds, no ms)    → "07:44:18:000 PM"
 */
function formatTimeOnly(timeStr) {
    if (!timeStr) return '-- : --';

    // New full format: HH:MM:SS:MS AM/PM  (e.g. "07:44:18:253 PM")
    const fullMatch = timeStr.match(/(\d{1,2}):(\d{2}):(\d{2}):(\d{1,3})\s*(AM|PM)/i);
    if (fullMatch) {
        return `${fullMatch[1].padStart(2,'0')}:${fullMatch[2]}:${fullMatch[3]}:${fullMatch[4].padStart(3,'0')} ${fullMatch[5].toUpperCase()}`;
    }

    // Seconds but no MS, with AM/PM  (e.g. "07:44:18 PM")
    const secMatch = timeStr.match(/(\d{1,2}):(\d{2}):(\d{2})\s*(AM|PM)/i);
    if (secMatch) {
        return `${secMatch[1].padStart(2,'0')}:${secMatch[2]}:${secMatch[3]}:000 ${secMatch[4].toUpperCase()}`;
    }

    // Short HH:MM AM/PM  (e.g. "07:44 PM")
    const amPmMatch = timeStr.match(/(\d{1,2}):(\d{2})\s*(AM|PM)/i);
    if (amPmMatch) {
        return `${amPmMatch[1].padStart(2,'0')}:${amPmMatch[2]}:00:000 ${amPmMatch[3].toUpperCase()}`;
    }

    // Pure 24-hr HH:MM (no AM/PM) — convert to 12-hr with ms zeroed
    const rawMatch = timeStr.match(/(\d{1,2}):(\d{2})/);
    if (rawMatch) {
        let hours   = parseInt(rawMatch[1], 10);
        const mins  = rawMatch[2];
        const ampm  = hours >= 12 ? 'PM' : 'AM';
        hours = hours % 12 || 12;
        return `${String(hours).padStart(2, '0')}:${mins}:00:000 ${ampm}`;
    }

    return timeStr; // fallback
}

/**
 * Utility to generate an aesthetic status badge depending on vehicle state.
 */
function renderStatusBadge(status) {
    if (status === 'IN') {
        return `<span class="px-2.5 py-1 bg-green-500/10 text-green-400 border border-green-500/40 rounded shadow-[0_0_8px_rgba(34,197,94,0.3)] text-xs font-bold tracking-wider rounded-md">
                    <i class="fa-solid fa-arrow-right-to-bracket mr-1.5"></i>IN
                </span>`;
    } else {
        return `<span class="px-2.5 py-1 bg-gray-600/20 text-gray-400 border border-gray-500/40 rounded text-xs font-bold tracking-wider rounded-md">
                    <i class="fa-solid fa-arrow-right-from-bracket mr-1.5"></i>OUT
                </span>`;
    }
}

/**
 * Utility to generate the respective icon for a vehicle type.
 */
function renderTypeIcon(type) {
    let baseClasses = "w-7 h-7 rounded flex items-center justify-center shrink-0 border shadow-md";
    if (type === 'Bus') {
        return `<div class="${baseClasses} bg-blue-500/20 text-blue-400 border-blue-500/30">
                    <i class="fa-solid fa-bus text-xs"></i>
                </div>`;
    } else if (type === 'Unknown') {
        // Red hue for unknown entities to alert security
        return `<div class="${baseClasses} bg-red-500/20 text-red-500 border-red-500/30">
                    <i class="fa-solid fa-car-side text-xs"></i>
                </div>`;
    } else {
        // Default styling for standard Cars or otherwise recognized
        return `<div class="${baseClasses} bg-gray-500/20 text-gray-300 border-gray-500/30">
                    <i class="fa-solid fa-car text-xs"></i>
                </div>`;
    }
}

/**
 * Insert a new vehicle object into the live table list, showcasing the slide-in animation.
 * @param {Object} vehicle - Required { type, plate, entryTime, exitTime, status }
 */
function publishVehicleEntry(vehicle) {
    const tableRow = document.createElement('tr');
    // Ensure styles are set including animation hook 'row-slide-in'
    tableRow.className = 'border-b border-white/5 row-slide-in hover:bg-white/5 transition-colors cursor-default text-center';
    
    tableRow.innerHTML = `
        <td class="py-1.5 px-4 text-left">
            <div class="flex items-center gap-3 text-[13px] justify-center">
                ${renderTypeIcon(vehicle.type)}
                <span class="font-medium ${vehicle.type === 'Unknown' ? 'text-red-400 font-bold' : 'text-gray-200'}">${vehicle.type}</span>
            </div>
        </td>
        <td class="py-1.5 px-4">
            <span class="font-mono bg-black/40 px-2 py-0.5 rounded text-white tracking-wider border border-white/10 shadow-inner text-[13px]">
                ${vehicle.plate}
            </span>
        </td>
        <td class="py-1.5 px-4 text-gray-300 font-medium text-[13px]">${formatTimeOnly(vehicle.entryTime)}</td>
        <td class="py-1.5 px-4 text-gray-400 text-[13px]">${vehicle.exitTime ? `<span class="text-gray-400 text-[13px]">${formatTimeOnly(vehicle.exitTime)}</span>` : `<span class="opacity-50">-- : --</span>`}</td>
        <td class="py-1.5 px-4 text-right text-[13px]"><div class="flex justify-center">${renderStatusBadge(vehicle.status)}</div></td>
        <td class="py-1.5 px-4 text-center">
            <button data-plate="${vehicle.plate}" data-status="${vehicle.status}"
                onclick="openEditModal('${vehicle.plate}', this.dataset.status)"
                class="inline-flex items-center gap-1.5 text-blue-300 hover:text-white bg-blue-600/20 hover:bg-blue-600/40 border border-blue-500/40 hover:border-blue-400 px-2.5 py-1 rounded-md text-[11px] font-semibold tracking-wide transition-all duration-200 shadow-sm hover:shadow-[0_0_8px_rgba(96,165,250,0.4)] focus:outline-none"
                title="Edit Entry">
                <i class="fa-solid fa-pen-to-square text-[10px]"></i> Edit
            </button>
        </td>
    `;
    
    // Inject at the very top
    if (tableBody.firstChild) {
        tableBody.insertBefore(tableRow, tableBody.firstChild);
    } else {
        tableBody.appendChild(tableRow);
    }

    // Auto-prune old records if over limit to keep performance high
    if (tableBody.children.length > 50) {
        tableBody.removeChild(tableBody.lastChild);
    }
}

function updateVehicleExit(vehicle) {
    const rows = tableBody.querySelectorAll('tr');
    for (let row of rows) {
        if (row.cells[1].textContent.includes(vehicle.plate)) {
            // Update exit time and status badge regardless of current status
            // (handles both IN→OUT and OUT exit-time corrections)
            row.cells[3].innerHTML = `<span class="text-gray-400 text-[13px]">${formatTimeOnly(vehicle.exitTime)}</span>`;
            row.cells[4].innerHTML = `<div class="flex justify-center">${renderStatusBadge(vehicle.status)}</div>`;
            // Sync edit button status so modal reflects the current state
            const editBtn = row.cells[5] && row.cells[5].querySelector('button[data-plate]');
            if (editBtn) editBtn.dataset.status = 'OUT';
            // Move this row to the top — most recently changed always appears first
            tableBody.insertBefore(row, tableBody.firstChild);
            if (typeof processExitMetrics === 'function') {
                processExitMetrics();
            } else if (typeof processExitChart === 'function') {
                processExitChart();
            }
            break; // Found and updated
        }
    }
}

window.deleteVehicleEntry = async function(plate) {
    if (!confirm(`Are you sure you want to delete the entry for ${plate}?`)) return;
    try {
        const hostname = location.hostname || '127.0.0.1';
        const response = await fetch(`http://${hostname}:8000/delete_entry/${plate}`, {
            method: 'DELETE'
        });
        if (response.ok) {
            console.log(`Successfully requested deletion for ${plate}`);
        } else {
            console.error("Failed to delete the entry");
        }
    } catch (err) {
        console.error("Error deleting entry:", err);
    }
};

// Ensure the table starts empty per requirements until backend connects
if (tableBody) {
    tableBody.innerHTML = '';
}

// Connect WebSocket for real-time AI updates
function initializeWebSocket() {
    try {
        const hostname = location.hostname || '127.0.0.1';
        const ws = new WebSocket(`ws://${hostname}:8000/ws`);
        
        ws.onmessage = (event) => {
            try {
                const message = JSON.parse(event.data);
                
                if (message.type === 'init') {
                    if (tableBody) tableBody.innerHTML = '';
                    if (Array.isArray(message.data)) {
                        message.data.reverse().forEach(publishVehicleEntry);
                    }
                    if (message.counters) updateCounters(message.counters);
                } else if (message.type === 'entry') {
                    publishVehicleEntry(message.data);
                    if (message.counters) updateCounters(message.counters);
                    processEntryChart();
                    triggerDetectionIndicator();
                } else if (message.type === 'exit') {
                    updateVehicleExit(message.data);
                    if (message.counters) updateCounters(message.counters);
                    processExitChart();
                    triggerDetectionIndicator();
                } else if (message.type === 'delete') {
                    const rows = tableBody.querySelectorAll('tr');
                    for (let row of rows) {
                        if (row.cells[1].textContent.includes(message.plate)) {
                            row.remove();
                            break;
                        }
                    }
                    if (message.counters) updateCounters(message.counters);
                }
            } catch (err) {
                console.error("WebSocket message parsing error:", err);
            }
        };
        
        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
        };
        
        ws.onclose = () => {
            console.log("WebSocket Connection closed. Retrying in 5 seconds...");
            setTimeout(initializeWebSocket, 5000);
        };
    } catch (err) {
        console.error("Failed to setup WebSocket:", err);
        setTimeout(initializeWebSocket, 5000);
    }
}
// Start real-time stream
initializeWebSocket();

/**
 * System Statistics Metrics Handlers
 */
const busCountElem = document.getElementById('bus-count');
const unknownCountElem = document.getElementById('unknown-count');
const totalCountElem = document.getElementById('total-count');

// Initialize to 0 on startup to clear old defaults
if (busCountElem) busCountElem.textContent = "0";
if (unknownCountElem) unknownCountElem.textContent = "0";
if (totalCountElem) totalCountElem.textContent = "0";

function updateCounters(counters) {
    if (totalCountElem) totalCountElem.textContent = counters.total_entries;
    if (busCountElem) busCountElem.textContent = counters.buses_inside;
    if (unknownCountElem) unknownCountElem.textContent = counters.unknown_vehicles;
}

function processEntryChart() {
    if (typeof trafficChart !== 'undefined' && trafficChart) {
        const lastIndex = trafficChart.data.datasets[0].data.length - 1;
        trafficChart.data.datasets[0].data[lastIndex] += 1;
        trafficChart.update();
    }
}

/**
 * Green Detection Indicator — blinks for 3s when a vehicle plate is detected,
 * then returns to steady green automatically.
 */
let _detectionIndicatorTimer = null;

function triggerDetectionIndicator() {
    const dot = document.getElementById('detection-dot');
    if (!dot) return;

    // Reset any existing timer so multiple detections extend the blink
    if (_detectionIndicatorTimer) {
        clearTimeout(_detectionIndicatorTimer);
    }

    dot.classList.add('detection-active');

    _detectionIndicatorTimer = setTimeout(() => {
        dot.classList.remove('detection-active');
        _detectionIndicatorTimer = null;
    }, 3000);
}

function processExitChart() {
    if (typeof trafficChart !== 'undefined' && trafficChart) {
        const lastIndex = trafficChart.data.datasets[1].data.length - 1;
        trafficChart.data.datasets[1].data[lastIndex] += 1;
        trafficChart.update();
    }
}


/**
 * System Status Monitoring logic
 */
function updateSystemStatus() {
    const indicator = document.getElementById('system-status-indicator');
    if (!indicator) return;

    if (navigator.onLine) {
        indicator.innerHTML = `System Status: <span class="text-green-400 font-semibold"><i class="fa-solid fa-circle-check"></i> Active</span>`;
    } else {
        indicator.innerHTML = `System Status: <span class="text-red-500 font-semibold"><i class="fa-solid fa-circle-xmark"></i> Not Active</span>`;
    }
}

// Initial check and listeners for network status
updateSystemStatus();
window.addEventListener('online', updateSystemStatus);
window.addEventListener('offline', updateSystemStatus);

/**
 * Camera Control Logic Factory
 */
function applyWebcamLogic(prefix) {
    const sfx = prefix ? `-${prefix}` : '';
    const btnStart = document.getElementById(`btn-start-capture${sfx}`);
    const btnStop = document.getElementById(`btn-stop-capture${sfx}`);
    
    // Determine the ID dynamically. Legacy '' maps to exit.
    const recId = prefix === 'entry' ? 'rec_status_entry' : 'rec_status_exit';
    const recInd = document.getElementById(recId);
    
    const stoppedPlc = document.getElementById(`stopped-placeholder${sfx}`);
    const videoElem = document.getElementById(`security-camera-feed${sfx}`);
    const initPlc = document.getElementById(`camera-placeholder${sfx}`);
    
    if (btnStart && btnStop) {
        btnStart.addEventListener('click', async () => {
            const hostname = location.hostname || '127.0.0.1';
            const apiEndpoint = prefix === 'entry' ? '/start_capture_entry' : '/start_capture_exit';
            
            try {
                // Signal backend to open camera resource
                await fetch(`http://${hostname}:8000${apiEndpoint}`);
            } catch (err) {
                console.error("Failed to start backend capture:", err);
            }

            // Connect to Python FastAPI Backend stream independently
            if (videoElem) {
                // Determine which endpoint to use based on the gate prefix
                const endpoint = prefix === 'entry' ? '/video_feed_entry' : '/video_feed_exit';
                // Adding timestamp to bypass browser cache
                videoElem.src = `http://${hostname}:8000${endpoint}?t=${new Date().getTime()}`;
                videoElem.classList.remove('hidden');
            }
            
            if (initPlc) initPlc.classList.add('hidden');
            if (recInd) recInd.style.visibility = 'visible'; // Reveal new animated overlay
            if (stoppedPlc) stoppedPlc.classList.add('hidden');
        });

        btnStop.addEventListener('click', async () => {
            const hostname = location.hostname || '127.0.0.1';
            const apiEndpoint = prefix === 'entry' ? '/stop_capture_entry' : '/stop_capture_exit';
            
            try {
                // Signal backend to close camera resource
                await fetch(`http://${hostname}:8000${apiEndpoint}`);
            } catch (err) {
                console.error("Failed to stop backend capture:", err);
            }

            // Disconnect from stream independently
            if (videoElem) {
                videoElem.src = '';
                videoElem.classList.add('hidden');
            }
            
            // Toggle aesthetics
            if (recInd) recInd.style.visibility = 'hidden'; // Hide the widget via visibility
            if (stoppedPlc) stoppedPlc.classList.remove('hidden');
        });
    }
}

// Instantiate specific streams for Dual Layout
applyWebcamLogic('entry'); // Entry Gate Camera
applyWebcamLogic('');      // Exit Gate Camera (Default Legacy IDs)

/**
 * Collapsible Sidebar Logic
 */
const sidebar = document.getElementById('sidebar');
const btnOpenSidebar = document.getElementById('btn-open-sidebar');
const btnCloseSidebar = document.getElementById('btn-close-sidebar');

if (sidebar && btnOpenSidebar && btnCloseSidebar) {
    // Open Sidebar
    btnOpenSidebar.addEventListener('click', () => {
        sidebar.classList.remove('hidden');
        sidebar.classList.add('flex');
    });

    // Close Sidebar
    btnCloseSidebar.addEventListener('click', () => {
        sidebar.classList.remove('flex');
        sidebar.classList.add('hidden');
    });
}

/**
 * Theme Toggle Logic
 */
const themeToggleBtn = document.getElementById('theme-toggle-btn');
if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
        try {
            document.body.classList.toggle('light-mode');
            
            const icon = themeToggleBtn.querySelector('i');
            if (icon) {
                if (document.body.classList.contains('light-mode')) {
                    icon.classList.remove('fa-moon');
                    icon.classList.add('fa-sun');
                } else {
                    icon.classList.remove('fa-sun');
                    icon.classList.add('fa-moon');
                }
            }
        } catch (err) {
            console.error("Theme toggle error:", err);
        }
    });
}

/**
 * Search/Filter Table Logic
 */
const searchInput = document.getElementById('search-input');
if (searchInput) {
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase().trim();
        // tableBody is defined locally at the top of this script
        const rows = tableBody.querySelectorAll('tr'); 
        
        rows.forEach(row => {
            // First column contains Type, Second column contains Plate
            const typeText = row.cells[0].textContent.toLowerCase();
            const plateText = row.cells[1].textContent.toLowerCase();
            
            if (typeText.includes(searchTerm) || plateText.includes(searchTerm)) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    });
}

/**
 * Hourly Traffic Modal & Chart Logic
 */
let trafficChart = null;

function initTrafficChart() {
    const ctx = document.getElementById('trafficChart');
    if (!ctx) return;
    
    // Base data simulating traffic for the past 6 hours
    const labels = [];
    const incomingDataPts = [];
    const outgoingDataPts = [];
    const nowHour = new Date().getHours();
    
    for (let i = 5; i >= 0; i--) {
        let h = nowHour - i;
        if (h < 0) h += 24;
        
        let ampm = h >= 12 ? 'PM' : 'AM';
        let displayHour = h % 12;
        displayHour = displayHour ? displayHour : 12; // the hour '0' should be '12'
        
        labels.push(`${displayHour} ${ampm}`);
        incomingDataPts.push(Math.floor(Math.random() * 40) + 10);
        outgoingDataPts.push(Math.floor(Math.random() * 30) + 5);
    }
    
    trafficChart = new window.Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Incoming Vehicles',
                    data: incomingDataPts,
                    borderColor: '#22c55e', // Neon green
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    borderWidth: 3,
                    pointBackgroundColor: '#4ade80',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#4ade80',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.4
                },
                {
                    label: 'Outgoing Vehicles',
                    data: outgoingDataPts,
                    borderColor: '#ef4444', // Neon red/pink
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 3,
                    pointBackgroundColor: '#f87171',
                    pointBorderColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#f87171',
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        plugins: [{
            id: 'glowPlugin',
            beforeDatasetDraw: function(chart, args) {
                const ctx = chart.ctx;
                ctx.save();
                const dataset = chart.data.datasets[args.index];
                ctx.shadowColor = dataset.borderColor;
                ctx.shadowBlur = 10;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;
            },
            afterDatasetDraw: function(chart) {
                chart.ctx.restore();
            }
        }],
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { 
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#e2e8f0',
                        font: { size: 13, weight: 'bold' }
                    }
                },
                tooltip: { 
                    mode: 'index', 
                    intersect: false,
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    titleColor: '#60a5fa',
                    bodyColor: '#e2e8f0',
                    borderColor: 'rgba(59, 130, 246, 0.3)',
                    borderWidth: 1
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Time (Hourly)', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    title: { display: true, text: 'Vehicle Count', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' },
                    beginAtZero: true
                }
            }
        }
    });
}

const btnHourlyTraffic = document.getElementById('btn-hourly-traffic');
const modalHourlyTraffic = document.getElementById('hourly-traffic-modal');
const btnCloseModal = document.getElementById('btn-close-modal');

if (btnHourlyTraffic && modalHourlyTraffic && btnCloseModal) {
    btnHourlyTraffic.addEventListener('click', () => {
        modalHourlyTraffic.classList.remove('hidden');
        if (!trafficChart) {
            // Slight delay ensures the modal display transitions don't mess up canvas sizing bounds
            setTimeout(initTrafficChart, 50); 
        } else {
            trafficChart.update();
        }
    });

    btnCloseModal.addEventListener('click', () => {
        modalHourlyTraffic.classList.add('hidden');
    });
}

/**
 * Manual Entry Modal Logic
 */
const btnManualEntry       = document.getElementById('btn-manual-entry');
const manualEntryModal     = document.getElementById('manual-entry-modal');
const btnCloseManualModal  = document.getElementById('btn-close-manual-modal');
const manualEntryForm      = document.getElementById('manual-entry-form');
const manualPlateInput     = document.getElementById('manual-plate');
const manualTimeInput      = document.getElementById('manual-time');
const manualTypeSelect     = document.getElementById('manual-type');

// Open modal
if (btnManualEntry && manualEntryModal) {
    btnManualEntry.addEventListener('click', () => {
        // Pre-fill current time with HH:MM:SS:MS
        const now = new Date();
        const hh  = String(now.getHours()).padStart(2, '0');
        const mm  = String(now.getMinutes()).padStart(2, '0');
        // time input only accepts HH:MM
        if (manualTimeInput) manualTimeInput.value = `${hh}:${mm}`;
        manualEntryModal.classList.remove('hidden');
    });
}

// Close modal
if (btnCloseManualModal && manualEntryModal) {
    btnCloseManualModal.addEventListener('click', () => {
        manualEntryModal.classList.add('hidden');
        if (manualEntryForm) manualEntryForm.reset();
    });
    // Close on backdrop click
    manualEntryModal.addEventListener('click', (e) => {
        if (e.target === manualEntryModal) {
            manualEntryModal.classList.add('hidden');
            if (manualEntryForm) manualEntryForm.reset();
        }
    });
}

// Force uppercase on every keystroke for plate input
if (manualPlateInput) {
    manualPlateInput.addEventListener('input', function () {
        const pos = this.selectionStart;
        this.value = this.value.toUpperCase();
        this.setSelectionRange(pos, pos);
    });
}

// Form submission
if (manualEntryForm) {
    manualEntryForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const plate     = manualPlateInput ? manualPlateInput.value.trim().toUpperCase() : '';
        const vtype     = manualTypeSelect ? manualTypeSelect.value : '';
        const timeVal   = manualTimeInput  ? manualTimeInput.value  : '';   // "HH:MM"

        if (!plate || !vtype || !timeVal) return;

        const submitBtn = document.getElementById('btn-submit-manual');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Adding...';
        }

        try {
            const hostname = location.hostname || '127.0.0.1';
            const response = await fetch(`http://${hostname}:8000/manual_entry`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: vtype, plate, entryTime: timeVal })
            });

            if (response.ok) {
                manualEntryModal.classList.add('hidden');
                manualEntryForm.reset();
                triggerDetectionIndicator();
            } else {
                const err = await response.json();
                alert(`Error: ${err.detail || 'Failed to add entry.'}`);
            }
        } catch (err) {
            console.error('Manual entry error:', err);
            alert('Could not connect to the backend. Please ensure the server is running.');
        } finally {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.innerHTML = '<i class="fa-solid fa-plus"></i> Add Entry';
            }
        }
    });
}

/**
 * Edit Entry Modal Logic
 * Shows two options: Delete Entry | Mark as OUT (with exit time input)
 */
let _editModalPlate  = '';
let _editModalStatus = '';

window.openEditModal = function(plate, status) {
    _editModalPlate  = plate;
    _editModalStatus = status;

    const modal           = document.getElementById('edit-entry-modal');
    const plateDisplay    = document.getElementById('edit-modal-plate');
    const markOutBtn      = document.getElementById('btn-modal-mark-out');
    const markOutSubform  = document.getElementById('edit-mark-out-subform');

    if (plateDisplay) plateDisplay.textContent = plate;

    // Reset sub-form to hidden state every time modal opens
    if (markOutSubform) markOutSubform.classList.add('hidden');
    // Always ensure the mark-out button is visible on modal open
    if (markOutBtn) markOutBtn.classList.remove('hidden');

    // Pre-fill current time in exit time input
    const now = new Date();
    const exitTimeInput = document.getElementById('edit-exit-time');
    if (exitTimeInput) {
        exitTimeInput.value = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;
    }

    // Update "Mark as OUT" button label based on vehicle status
    if (markOutBtn) {
        markOutBtn.disabled = false;
        markOutBtn.classList.remove('opacity-40', 'cursor-not-allowed');
        if (status === 'OUT') {
            // Allow editing exit time even if already OUT
            markOutBtn.innerHTML = `<i class="fa-solid fa-clock-rotate-left w-4 text-center"></i><span>Edit Exit Time</span><span class="ml-auto text-[10px] text-gray-500 uppercase tracking-wider">Update</span>`;
            markOutBtn.title = 'Click to edit the recorded exit time';
        } else {
            markOutBtn.innerHTML = `<i class="fa-solid fa-arrow-right-from-bracket w-4 text-center"></i><span>Mark as OUT</span><span class="ml-auto text-[10px] text-gray-500 uppercase tracking-wider">Set Exit Time</span>`;
            markOutBtn.title = '';
        }
    }

    if (modal) modal.classList.remove('hidden');
};

// Close edit modal
const editModal         = document.getElementById('edit-entry-modal');
const btnCloseEditModal = document.getElementById('btn-close-edit-modal');

if (btnCloseEditModal && editModal) {
    btnCloseEditModal.addEventListener('click', () => {
        editModal.classList.add('hidden');
    });
    editModal.addEventListener('click', (e) => {
        if (e.target === editModal) editModal.classList.add('hidden');
    });
}

// Delete from edit modal
const btnModalDelete = document.getElementById('btn-modal-delete');
if (btnModalDelete) {
    btnModalDelete.addEventListener('click', async () => {
        editModal.classList.add('hidden');
        // Re-use existing deleteVehicleEntry which includes the confirm dialog
        await window.deleteVehicleEntry(_editModalPlate);
    });
}

// Show Mark as OUT sub-form — start live running clock when opened
const btnModalMarkOut    = document.getElementById('btn-modal-mark-out');
const markOutSubform     = document.getElementById('edit-mark-out-subform');
let _markOutClockInterval = null;

function _stopMarkOutClock() {
    if (_markOutClockInterval) {
        clearInterval(_markOutClockInterval);
        _markOutClockInterval = null;
    }
}

if (btnModalMarkOut && markOutSubform) {
    btnModalMarkOut.addEventListener('click', () => {
        markOutSubform.classList.remove('hidden');

        // Stop any existing clock before starting fresh
        _stopMarkOutClock();

        // Live running clock — updates every 100ms for HH:MM:SS:MS accuracy
        function tickClock() {
            const t = buildTimestamp(new Date());
            // Display the running time in the live-clock span (created below)
            const clockSpan = document.getElementById('mark-out-live-clock');
            if (clockSpan) clockSpan.textContent = t;
        }
        tickClock(); // immediate first tick
        _markOutClockInterval = setInterval(tickClock, 100);

        // Scroll into view for convenience
        const exitInput = document.getElementById('edit-exit-time');
        if (exitInput && !exitInput.value) {
            const n = new Date();
            exitInput.value = `${String(n.getHours()).padStart(2,'0')}:${String(n.getMinutes()).padStart(2,'0')}`;
        }
    });
}

// Confirm Mark as OUT — capture exact live timestamp at click moment
const btnConfirmMarkOut = document.getElementById('btn-confirm-mark-out');
if (btnConfirmMarkOut) {
    btnConfirmMarkOut.addEventListener('click', async () => {
        // Stop the running clock immediately on click
        _stopMarkOutClock();

        // Build the precise exit timestamp at the exact moment of click
        const exactTimestamp = buildTimestamp(new Date());

        // We still need a HH:MM value for the backend /mark_exit endpoint
        // (backend converts HH:MM → HH:MM:SS:000 — but here we have full precision)
        // Send raw 24-hr HH:MM as the payload (backend will zero-pad SS and MS)
        const now = new Date();
        const exitTimeHHMM = `${String(now.getHours()).padStart(2,'0')}:${String(now.getMinutes()).padStart(2,'0')}`;

        // Also display the precise ms timestamp in the table by injecting it into vehicle_db
        // via a custom header so the broadcast picks it up  — instead, we send it as-is and
        // let the backend produce HH:MM:SS:000.  The seconds/ms from this click are already
        // captured because the server records datetime.datetime.now() on its side with ms.
        // So just send the standard HH:MM value and the backend will produce a fresh now_str
        // with full ms precision automatically.
        const exitTime = exitTimeHHMM;

        if (!exitTime) { alert('Please enter a valid exit time.'); return; }

        btnConfirmMarkOut.disabled = true;
        btnConfirmMarkOut.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving...';

        try {
            const hostname = location.hostname || '127.0.0.1';
            const response = await fetch(`http://${hostname}:8000/mark_exit`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ plate: _editModalPlate, exitTime })
            });

            if (response.ok) {
                editModal.classList.add('hidden');
                triggerDetectionIndicator();
            } else {
                const err = await response.json();
                alert(`Error: ${err.detail || 'Failed to mark as OUT.'}`);
            }
        } catch (err) {
            console.error('Mark exit error:', err);
            alert('Could not connect to the backend. Please ensure the server is running.');
        } finally {
            btnConfirmMarkOut.disabled = false;
            btnConfirmMarkOut.innerHTML = '<i class="fa-solid fa-check"></i> Confirm OUT';
        }
    });
}

// Stop clock when edit modal is closed without confirming
if (btnCloseEditModal) {
    const _origClose = btnCloseEditModal.onclick;
    btnCloseEditModal.addEventListener('click', _stopMarkOutClock);
}
if (editModal) {
    editModal.addEventListener('click', (e) => {
        if (e.target === editModal) _stopMarkOutClock();
    });
}
